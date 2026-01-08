#!/usr/bin/env python3

import re
import os
import sys
import yaml
import numpy as np

import threading
import time

import xml.etree.ElementTree as ET
import tempfile

import adios2
from effis.composition import EffisLogger as Logger


class VariableRemap:

    InName = None
    OutName = None

    @classmethod
    def StrToDict(cls, instr):
        if not isinstance(instr, str):
            raise TypeError(f"Must give string to StrToDict. Given type is {type(instr)}")
        instr = instr.strip()
        if len(instr) == 0:
            raise NameError("Empty string doesn't make sense for StrToDict")

        pattern = (
            r"^"
            r"(?P<varname>[^\[\]\(\)\s]+?)"     # Name (don't look for parentheses/brackets like this)
            r"\s*"
            r"(?P<transpose>\([\s\d,]*\))?"     # Transposes
            r"\s*"
            r"(?P<selection>\[[\s\d,:]*\])?"    # Selections
            r"$"
        )
        c = re.compile(pattern)
        result = c.search(instr)

        if result is None:
            raise NameError(
                f"{instr} is not an appropriate remapping string. "
                f"Use varname(transposes)[selections]. "
                f"Only integers can be used in tranposes and selections. "
                f"If [], (), or whitespace are needed in varname, use keyword formatting."
            )

        ReturnDict = {
            'InName': result.group("varname"),
            'Transpose': cls.GetInside(result.group("transpose")),
            'Selection': cls.GetInside(result.group("selection")),
        }
        return ReturnDict


    @staticmethod
    def ArrIntFind(instr, label=None):
        if label is None:
            label = instr
        arr = instr.split(",")
        pattern = re.compile(r"[\[\(\{\s]*(\d+)[\]\)\}\s]*")
        for i, dim in enumerate(arr):
            dim = dim.strip()
            result = pattern.search(dim)
            if result is not None:
                arr[i] = result.group(1)
            else:
                raise IndexError(f"Invalid selection dimension: {dim} from {label}")
        return arr


    @staticmethod
    def GetInside(result):
        if result is None:
            return None
        else:
            s = result[1:-1].strip()
            pattern = re.compile(r"\s")
            s = pattern.sub("", s)
            return s


    def __init__(
        self,
        OutName=None,
        InName=None,
        Transpose=None,
        Selection=None,
    ):
        self.OutName = OutName
        self.InName = InName

        if Transpose is None:
            self.Transpose = []
        elif isinstance(Transpose, (tuple, list, np.ndarray)):
            self.Transpose = list(Transpose)
        elif isinstance(Transpose, str):
            self.Transpose = self.ArrIntFind(Transpose)
        else:
            raise TypeError("Unknown Transpose type")

        for i in range(len(self.Transpose)):
            try:
                self.Transpose[i] = int(self.Transpose[i])
            except Exception:
                raise TypeError("Transpose elements must be integers.")
        self.Transpose = tuple(self.Transpose)
        if len(np.unique(self.Transpose)) != len(self.Transpose):
            raise IndexError(
                f"Invalid transpose: {self.InName}{self.Transpose}. "
                f"Cannot repeate dimension."
            )

        self.SelectionStr = Selection
        if isinstance(self.SelectionStr, list):
            self.SelectionStr = str(self.SelectionStr)


def ExistsVerify(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError("{0} not found".format(filename))


def GenerateListing(filename, write=None):
    data = {}
    ExistsVerify(filename)

    with adios2.FileReader(filename) as stream:

        variables = stream.available_variables()

        for varname in variables:

            # Attributes will be needed as well
            var = stream.inquire_variable(varname)

            # Numpy doesn't recognize _t types, but keep like ADIOS for now
            t = var.type()
            if t.endswith("_t"):
                t = t[:-2]

            shape = var.shape()

            data[varname] = {
                #'dtype': np.dtype(t),
                #'shape': shape,
                'dtype': var.type(),
                'ndim': len(shape),
                'steps': var.steps(),
            }

    if write is not None:
        with open(write, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False)

    return data


def FileReplicate(filename):

    data = GenerateListing(filename)

    outmap = {}
    for varname in data:
        outmap[varname] = varname

    server = BlockServer(outmap)
    server.AddDataFile(filename)
    server.DoIO(filename.replace(".bp", ".copy.bp"))


class MPISetup:

    ANY_SOURCE = None

    @staticmethod
    def Get_rank():
        return 0

    @staticmethod
    def Get_size():
        return 1

    @staticmethod
    def Bcast(*args, **kwargs):
        pass

    @staticmethod
    def Send(*args, **kwargs):
        pass

    @classmethod
    def GetMPI(cls, forceoff=False):
        if forceoff:
            return cls()

        try:
            from mpi4py import MPI
        except Exception:
            return cls()
        else:
            cls.ANY_SOURCE = MPI.ANY_SOURCE
            return MPI.COMM_WORLD
        

class BlockServer:
   
    DataFiles = []

    # These are only used on Rank 0
    MapInput = None
    FileSpecify = {}

    @staticmethod
    def DictOrYAML(invar):
        if isinstance(invar, dict):
            return invar
        else:
            ExistsVerify(invar)
            with open(invar, 'r') as infile:
                data = yaml.safe_load(infile)
            return data

    def AddDataFile(self, filename):
        ExistsVerify(filename)
        filename = os.path.abspath(filename)
        if filename not in self.DataFiles:
            self.DataFiles += [filename]

    def Specify(self, filename):
        if self.rank == 0:
            self.FileSpecify = self.DictOrYAML(filename)
            for varname in self.FileSpecify:
                self.FileSpecify[varname] = os.path.abspath(self.FileSpecify[varname])

    def __init__(self, MapFile):
        self.comm = MPISetup.GetMPI(forceoff=False)
        self.rank = self.comm.Get_rank()
        self.MapInput = MapFile


    def _BuildVR(self):

        self.MapDict = self.DictOrYAML(self.MapInput)

        VarInfo = {}
        AllVars = []
        for datafile in self.DataFiles:
            VarInfo[datafile] = GenerateListing(datafile)
            AllVars += list(VarInfo[datafile].keys())

        self.vrs = {}
        for outname in self.MapDict:

            if isinstance(self.MapDict[outname], str):
                vdict = VariableRemap.StrToDict(self.MapDict[outname])
                vr = VariableRemap(OutName=outname, **vdict)
            else:
                vr = VariableRemap(OutName=outname, **self.MapDict[outname])

            #vr = VariableRemap(outname, self.MapDict[outname])

            Logger.Info(
                f"{vr.OutName} <-- {vr.InName}; "
                f"transpose: {vr.Transpose}, "
                f"selection: {vr.SelectionStr}"
            )

            if vr.InName in self.FileSpecify:
                filestr = self.FileSpecify[vr.InName]
                Vars = list(VarInfo[filestr].keys())
                vr.InFilename = filestr
            else:
                Vars = AllVars
                filestr = ", ".join(self.DataFiles)

            if vr.InName not in Vars:
                raise NameError(
                    f"{vr.InName} is not a variable found in {filestr}."
                )
            elif Vars.count(vr.InName) > 1:
                raise NameError(
                    f"{vr.InName} occurs in multiple files. Need to Specify which."
                )

            if "InFilename" not in dir(vr):
                for datafile in VarInfo:
                    if vr.InName in VarInfo[datafile]:
                        vr.InFilename = datafile
                        break

            if vr.InFilename not in self.vrs:
                self.vrs[vr.InFilename] = []

            self.vrs[vr.InFilename] += [vr]


    def BuildVR(self):

        if self.rank == 0:
            self._BuildVR()
        else:
            self.vrs = None
        self.vrs = self.comm.bcast(self.vrs, root=0)
        # self._BuildVR()

        UsedFilenames = list(filename for filename in self.DataFiles if filename in self.vrs)

        return UsedFilenames


    @staticmethod
    def GetAsInts(arr):
        outarr = np.empty(len(arr), dtype=np.int64)
        for i, val in enumerate(arr):
            outarr[i] = int(val)
        return outarr

    @staticmethod
    def GetTransposeDims(vr, var):
        shape = var.shape()
        ndim = len(shape)

        if len(vr.Transpose) == 0:
            TransposeDims = tuple(range(ndim))
        else:
            if len(vr.Transpose) != ndim:
                raise IndexError(
                    f"Invalid transpose: {var.name()}{vr.Transpose}. "
                    f"Must be same number of dimensions as {var.name()} ({ndim})"
                )
            TransposeDims = vr.Transpose

        return TransposeDims

    @staticmethod
    def BlankOrInt(group, value=0, minus=0):
        if (group is None) or (len(group.strip()) == 0):
            return value
        else:
            return int(group) - minus


    @staticmethod
    def WriteXML(varname, filename, Start, Count):

        if len(Start) > 0:

            root = ET.Element("adios-query")
            io = ET.SubElement(root, "io", name=filename)
            var = ET.SubElement(io, "var", name=varname)

            s = []
            c = []
            for i in range(len(Start)):
                s += [str(Start[i])]
                c += [str(Count[i])]
            bb = ET.SubElement(var, "boundingbox", start=','.join(s), count=','.join(c))


            ET.indent(root, space="  ")
            xml_string = ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')

            with tempfile.NamedTemporaryFile(
                mode='w',
                delete=False,
                suffix=".xml",
                prefix=f"{varname}-"
            ) as outfile:
                outfile.write(xml_string)

            return outfile.name

        else:
            '''
            # Haven't figured out how to Query writeblock, if possible
            wb = ET.SubElement(var, "writeblock")
            op = ET.SubElement(wb, "op", value="OR")
            ge = ET.SubElement(op, "range", compare="GE", value="0")
            '''
            return None


    @classmethod
    def GetSelectionDims(cls, vr, var):
        shape = var.shape()
        ndim = len(shape)

        Start = [0]*ndim
        Count = var.shape()

        if vr.SelectionStr is not None:

            sarr = vr.SelectionStr.split(",")
            if len(sarr) > ndim:
                raise IndexError(
                    f"Invalid selection: {var.name()}{vr.Transpose}[{vr.SelectionStr}]. "
                    f"[...] dimensionality cannot be greater than {ndim}"
                )

            for i, sdim in enumerate(sarr):
                sdim = sdim.strip()

                pattern = re.compile(r"(\d*)\s*:\s*(\d*)")
                result = pattern.search(sdim)
                if result is not None:
                    Start[i] = cls.BlankOrInt(result.group(1), 0)
                    Count[i] = cls.BlankOrInt(result.group(2), Count[i]-Start[i], Start[i])
                else:
                    arr = vr.ArrIntFind(sdim, label=vr.SelectionStr)
                    Start[i] = int(arr[0])
                    Count[i] = 1

        return Start, Count


    def GetSelections(self, OpenStreams, ClosedStreams):
        fvb = np.empty((0, 3), dtype=np.int64)

        for i, filename in enumerate(self.OpenFilenames):

            for j, vr in enumerate(self.vrs[filename]):

                var = OpenStreams[filename].inquire_variable(vr.InName)
                if var is None:
                    continue

                TransposeDims = self.GetTransposeDims(vr, var)
                Start, Count = self.GetSelectionDims(vr, var)

                xml = self.WriteXML(
                    var.name(),
                    filename,
                    np.array(Start)[list(TransposeDims)],
                    np.array(Count)[list(TransposeDims)],
                )

                if xml is not None:
                    w = adios2.bindings.Query(xml, OpenStreams[filename].engine.impl)
                    #NeededBlocks = w.GetResult()
                    NeededBlocks = w.GetBlockIDs()
                    if os.path.exists(xml):
                        os.remove(xml)
                else:
                    blocks = OpenStreams[filename].engine.blocks_info(vr.InName, 0)
                    NeededBlocks = list(range(len(blocks)))

                arr = np.empty((len(NeededBlocks), 3), dtype=np.int64)
                arr[:, 0] = i
                arr[:, 1] = j
                arr[:, 2] = NeededBlocks

                fvb = np.append(fvb, arr, axis=0)

        return fvb


    def DoIO(self, outname):

        ClosedStreams = []
        OpenStreams = {}

        UsedFilenames = self.BuildVR()

        commargs = []
        if not isinstance(self.comm, MPISetup):
            commargs += [self.comm]
        adios = adios2.Adios(*commargs)

        # Open on all ranks
        for filename in UsedFilenames:
            Logger.Info(f"Opening {filename}")
            io = adios.declare_io(filename)
            OpenStreams[filename] = adios2.Stream(io, filename, "r", *commargs)

        io = adios.declare_io("out")
        OutStream = adios2.Stream(io, outname, "w", *commargs)

        while OpenStreams:

            # begin_step on all ranks
            for filename in OpenStreams:
                status = OpenStreams[filename].begin_step()
                if status == adios2.StepStatus.EndOfStream:
                    Logger.Info(f"Closing {filename}")
                    OpenStreams[filename].close()
                    ClosedStreams += [filename]

            self.OpenFilenames = []
            for i, filename in enumerate(
                filename for filename in OpenStreams if filename not in ClosedStreams
            ):
                self.OpenFilenames += [filename] 
                if i == 0:
                    OutStream.begin_step()

            # distribute on rank 0
            if self.rank == 0:

                # file #, var #, block #, (and figure the rest out from the YAML)
                fvb = self.GetSelections(OpenStreams, ClosedStreams)

                tid = threading.Thread(
                    target=self.ServeData,
                    kwargs={'fvb': fvb},
                )
                tid.start()

            # Do reads/writes here
            self.ReceiveData(OpenStreams, OutStream)

            # Not sure I'll need this
            if self.rank == 0:
                tid.join()

            # end_step on all ranks
            for i, filename in enumerate(self.OpenFilenames):
                OpenStreams[filename].end_step()
                if i == 0:
                    OutStream.end_step()

            for i, filename in enumerate(ClosedStreams):
                del OpenStreams[filename]
                if i + 1 == len(ClosedStreams):
                    ClosedStreams = []

        OutStream.close()


    def ServeData(self, fvb=None):

        for i in range(fvb.shape[0]):
           
            workrank = np.empty((1,), dtype=np.int32)

            self.comm.Recv(
                workrank,
                source=MPISetup.ANY_SOURCE,
                tag=1,
            )

            self.comm.Send(
                fvb[i, :],
                dest=workrank[0],
                tag=0,
            )

        for i in range(self.comm.Get_size()):

            self.comm.Send(
                np.array([-1, -1, -1], dtype=np.int64),
                dest=i,
                tag=0,
            )


    def ReceiveData(self, OpenStreams, OutStream):

        while True:

            workrank = np.array([self.rank], dtype=np.int32)
            self.comm.Send(
                workrank,
                dest=0,
                tag=1,
            )

            fvb = np.empty((3,), dtype=np.int64)
            self.comm.Recv(
                fvb,
                source=0,
                tag=0,
            )

            if fvb[0] == -1:
                break
            else:
                filenumber, varnumber, blocknumber = fvb
                filename = self.OpenFilenames[filenumber]
                varname = self.vrs[filename][varnumber].InName
                outname = self.vrs[filename][varnumber].OutName

                var = OpenStreams[filename].inquire_variable(varname)
                #var.set_block_selection(blocknumber)

                TransposeDims = self.GetTransposeDims(self.vrs[filename][varnumber], var)
                Start, Count = self.GetSelectionDims(self.vrs[filename][varnumber], var)
                start = np.array(Start)[list(TransposeDims)]
                count = np.array(Count)[list(TransposeDims)]

                block = OpenStreams[filename].engine.blocks_info(self.vrs[filename][varnumber].InName, 0)[blocknumber]
                blockstart = self.GetAsInts(block['Start'].split(','))[list(TransposeDims)]
                blockcount = self.GetAsInts(block['Count'].split(','))[list(TransposeDims)]
                thisstart = np.copy(blockstart)
                thiscount = np.copy(blockcount)

                if Count == var.shape():
                    data = OpenStreams[filename].read(varname, block_id=blocknumber)

                else:

                    for i in range(start.shape[0]):

                        if start[i] > blockstart[i]:
                            thisstart[i] = start[i]

                        if (thisstart[i] + count[i]) > (blockstart[i] + blockcount[i]):
                            diff = thisstart[i] - blockstart[i]
                            thiscount[i] = blockcount[i] - diff

                        if (thisstart[i] + count[i]) < (blockstart[i] + blockcount[i]):
                            thiscount[i] = count[i]

                    data = OpenStreams[filename].read(varname, start=list(thisstart), count=list(thiscount))
                
                data = np.transpose(data, axes=TransposeDims)
                newstart = thisstart[list(TransposeDims)].tolist()
                
                OutStream.write(outname, np.ascontiguousarray(data), Count, newstart, list(data.shape))



if __name__ == "__main__":

    filename = "/Users/eqs/Software/adios/test/gray-scott/gs.bp"
   
    #FileReplicate(filename)

    server = BlockServer("slice.yaml")
    server.AddDataFile(filename)
    specify = {'V': filename}
    server.Specify(specify)
    server.DoIO("test.bp")
