#!/usr/bin/env python3

import re
import os
import sys
import yaml
import numpy as np
import adios2


class AdiosBounds:

    GlobalSize = None
    Start = None
    LocalCount = None


class BlockInfo:

    pass


class VariableRemap:

    InName = None
    OutName = None

    DimStr = None
    TransposeStr = None
    SelectionStr = None


    @staticmethod
    def GetInside(result):
        if result is None:
            return None
        else:
            s = result[1:-1].strip()
            pattern = re.compile(r"\s")
            s = pattern.sub("", s)
            return s


    def SetDimStr(self, DimStr):

        if DimStr is None:
           DimStr = ""
        DimStr = DimStr.strip()

        if len(DimStr) > 0:
            pattern = (
                r"^\s*"
                r"[\'\"]?(?P<varname>.*?)[\'\"]?"
                r"(?P<transpose>\([\s\d,]*\))?"  # Transposes
                r"\s*"
                r"(?P<selection>\[[\s\d,:]*\])?" # Selections
                r"\s*$"
            )
            c = re.compile(pattern)
            result = c.search(DimStr)

            if result is None:
                raise IndexError(
                    "{0} is not an appropriate remapping. "
                    "Use (transposes)[selections]".format(DimStr)
                )
            else:
                DimStr = []
                self.InName = result.group("varname")

                t = self.GetInside(result.group("transpose"))
                if t is not None:
                    self.TransposeStr = t
                    DimStr += ["({0})".format(t)]

                s = self.GetInside(result.group("selection"))
                if s is not None:
                    self.SelectionStr = s
                    DimStr += ["[{0}]".format(s)]

        self.DimStr = "".join(DimStr)
        


    def __init__(
        self,
        OutName=None,
        DimStr=None,
    ):
        self.OutName = OutName
        self.SetDimStr(DimStr)


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
    server.BuildQueue()

    '''
    with adios2.Stream(filename, "r") as stream:

        for _ in stream.steps():

            #print(stream.current_step())
            variables = stream.available_variables()
            print(variables)
            for varname in variables:
                vr = VariableRemap(varname, varname)

                blocks = stream.engine.blocks_info(varname, 0)
                print(varname, blocks)
                break

                if vr.SelectionStr is None:
                    # Do All
                    pass
            break
    '''


class BlockServer:
   
    MapDict = None
    DataFiles = []
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
        self.FileSpecify = self.DictOrYAML(filename)
        for varname in self.FileSpecify:
            self.FileSpecify[varname] = os.path.abspath(self.FileSpecify[varname])

    def __init__(self, MapFile):
        self.MapDict = self.DictOrYAML(MapFile)


    def BuildQueue(self):

        VarInfo = {}
        for datafile in self.DataFiles:
            VarInfo[datafile] = GenerateListing(datafile)

        print('VarInfo:', VarInfo)
        print('MapDict:', self.MapDict)
        print('FileSpecify:', self.FileSpecify)

        for outname in self.MapDict:
            vr = VariableRemap(outname, self.MapDict[outname])
            print(
                vr.OutName, "-->", vr.InName, "",
                "transpose:", vr.TransposeStr, "",
                "selection:", vr.SelectionStr,
            )


if __name__ == "__main__":

    filename = "/Users/eqs/Software/adios/test/gray-scott/gs.bp"
   
    '''
    FileReplicate(filename)
    #print(GetTransposeSlice("slice.yaml"))
    '''

    server = BlockServer("slice.yaml")
    server.AddDataFile(filename)
    specify = {'V': filename}
    server.Specify(specify)
    server.BuildQueue()
