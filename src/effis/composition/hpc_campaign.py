import os
import shutil
import subprocess

from .log import CompositionLogger
from .workflow import Chdir

from hpc_campaign.manager import Manager


class Campaign:

    cmd = "hpc_campaign"

    @staticmethod
    def CheckString(value, label):
        if (value is not None) and (not isinstance(value, str)):
            CompositionLogger.RaiserError(
                ValueError,
                "Must give {1} as a string. Supplied {0}".format(value, label)
            )

    def __init__(
        self,
        filename=None,
        hostname=None,
        keyfile=None,
        create=False,
    ):
        if shutil.which(self.cmd) is None:
            CompositionLogger.RaiseError(
                "{0} not found. Cannot use Campaign".format(self.cmd)
            )
        self.CheckString(filename, "filename (path) Campaign initializer")
        self.CheckString(hostname, "hostname")
        self.CheckString(keyfile, "keyfile (path)")
        if (keyfile is not None) and (not os.path.isfile(keyfile)):
            CompositionLogger.RaiseError(
                ValueError,
                "Supplied {0} is not an existing file".format(keyfile)
            )

        self.filename = os.path.abspath(filename)
        self.hostname = hostname
        self.keyfile = keyfile
        self._create = True

        if os.path.isfile(self.filename):
            CompositionLogger.Info(
                "Found campaign {0}".format(self.filename)
            )
        elif create:
            CompositionLogger.Info(
                "Creating campaign {0}".format(self.filename)
            )
            self.Create()
        else:
            self._create = False

        self.manager = Manager(
            archive=self.filename,
            #campaign_store=str(campaign_store)
        )

    @property
    def _manager_(self):
        cmd = [self.cmd, "manager"]
        for name in ("hostname", "keyfile"):
            attr = getattr(self, name)
            if attr is not None:
                cmd += ["--{0}".format(name), attr]
        return cmd + [self.filename]

    def Manager(
        self,
        cmd,
        *args
    ):
        cmd = self._manager_ + [cmd, *args]
        subprocess.call(cmd)

    def Create(self):
        if not os.path.exists(self.filename):
            self.Manager("create")
        else:
            CompositionLogger.Warning(
                "{0} already exists. Skipping create.".format(self.filename)
            )


    def FileChecks(self, filename):

        if not os.path.isfile(self.filename) and self._create:
            CompositionLogger.Warning(
                "{0} does not exist. Skipping adding it.".format(
                    self.filename
                )
            )
            return False
        elif not os.path.exists(filename):
            CompositionLogger.Warning(
                "{0} does not exist. Skipping adding it".format(
                    filename
                )
            )
            return False

        return True

    def Data(self, filename, name=None, cli=False):
        self.CheckString(name, "name")
        if not self.FileChecks(filename):
            return

        if cli:
            dirpath = os.path.dirname(self.filename)
            relpath = os.path.relpath(os.path.abspath(filename), start=dirpath)
            args = ["data", relpath]
            if name is not None:
                args += ["--name", name]
            with Chdir(dirpath):
                self.Manager(*args)
        else:
            self.manager.data(filename, name=name)

    def Image(self, filename, name=None, cli=False):
        self.CheckString(name, "name")
        if not self.FileChecks(filename):
            return

        if cli:
            dirpath = os.path.dirname(self.filename)
            relpath = os.path.relpath(os.path.abspath(filename), start=dirpath)
            args = ["image", relpath]
            if name is not None:
                args += ["--name", name]
            with Chdir(dirpath):
                self.Manager(*args)
        else:
            self.manager.image(filename, name=name)

    def Text(self, filename, name=None, store=False, cli=False):
        self.CheckString(name, "name")
        if not self.FileChecks(filename):
            return

        if cli:
            dirpath = os.path.dirname(self.filename)
            relpath = os.path.relpath(os.path.abspath(filename), start=dirpath)
            args = ["text", relpath]
            if name is not None:
                args += ["--name", name]
            if store:
                args += ["--store"]
            with Chdir(dirpath):
                self.Manager(*args)
        else:
            self.manager.text(filename, name=name, store=store)

