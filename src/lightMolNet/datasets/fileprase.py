# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : fileprase.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import os
import re

from ase.units import Hartree, eV


class CalculationResultError(Exception):
    pass


class G16LogFiles(object):
    r"""
    .log File parser for g16 calculation result.

    Attributes
    ----------
    content:dict
        dict consist of all useful contents in the file.
    """

    def __init__(self, path: str, readOpt=True):
        self.path = path
        self.content = {'taskTitle': '',
                        'taskMethod': '',
                        'StructureDict': [],
                        'CPUTime': {
                            'days': None,
                            'hours': None,
                            'minutes': None,
                            'seconds': None
                        },
                        'NormalTerminate': False,
                        'EnergyDict': [],
                        'SpinMulti': 1,
                        'Charge': 0,
                        # 'GAPDict': {
                        #     'SCFAttr': [],
                        #     'SCFEnergy': []
                        #     }
                        }
        if readOpt:
            # read context from the file.
            self.readfiles()

    def readfiles(self):
        par_tasktitle = re.compile(
            r' -+\n (.*?)\n -+\n Symbolic Z-matrix:')
        par_tasktitle_restart = re.compile(
            r' -+\n (.*?)\n -+\n Structure from the checkpoint')
        par_taskmethod = re.compile(
            r' #(.*?)\n')
        par_cputime = re.compile(
            r'Job cpu time:(.*?) days (.*?) hours (.*?) minutes (.*?) seconds.\n', re.S)
        par_NormalTerminate = re.compile(r'Normal termination', re.S)
        par_ErrorTerminate = re.compile(r'Error termination', re.S)

        par_EnergyDictSCF = re.compile(r'SCF Done:.*?= {2}(.*?) {5}A.U.', re.S)
        par_StructureDictSCF = re.compile(
            r'Number {5}Number {7}Type {13}X {11}Y {11}Z\n '
            r'---------------------------------------------------------------------\n (.*?)\n '
            r'---------------------------------------------------------------------',
            re.S)
        par_SpinMulti = re.compile(r'Multiplicity = (.*?)\n')
        par_Charge = re.compile(r'Charge = (.*?) Multiplicity')
        """

        Parameters
        ----------
        mode:str
            if "all", read every "SCF Done" context.
            if "last", read the final result.
            # TODO: More read modes.

        Returns
        -------

        """
        with open(self.path) as file:
            str = file.read()
            self.content['taskMethod'] = par_taskmethod.findall(str)[0]
            try:
                self.content['taskTitle'] = par_tasktitle.findall(str)[0]
            except IndexError as e:
                try:
                    self.content['taskTitle'] = par_tasktitle_restart.findall(str)[0]
                except IndexError as e:
                    if "restart" in self.content['taskMethod']:
                        pass
                    else:
                        raise e
            try:
                self.content['CPUTime']['days'], self.content['CPUTime']['hours'], self.content['CPUTime']['minutes'], \
                self.content['CPUTime']['seconds'] = [float(i) for i in par_cputime.findall(str)[0]]
                self.content['NormalTerminate'] = (par_NormalTerminate.findall(str)[0]) != '' and (
                        len(par_ErrorTerminate.findall(str)) == 0)
            except:
                raise
            self.content['EnergyDict'] = [float(i) for i in par_EnergyDictSCF.findall(str)]
            self.content['StructureDict'] = [i for i in par_StructureDictSCF.findall(str)]
            self.content['SpinMulti'] = float(par_SpinMulti.findall(str)[0])
            self.content['Charge'] = float(par_Charge.findall(str)[0])

    def exportcontent(self):
        """

        Returns
        -------
        str:
            result summary

        """
        return self.path + "\t" + self.content['taskMethod'] + "\t" + str(
            self.content['EnergyDict'][-1]) + '\t' + str(self.content['NormalTerminate']) + '\n'

    def get_content(self, idx):
        r"""

        Parameters
        ----------
        idx:int
            index of which structure in the file.

        Returns
        -------

        """
        if self.content['NormalTerminate']:
            return {'taskTitle': self.content['taskTitle'],
                    'taskMethod': self.content['taskMethod'],
                    'StructureDict': self.content['StructureDict'][idx],
                    'EnergyDict': self.content['EnergyDict'][idx],
                    'SpinMulti': self.content['SpinMulti'],
                    'Charge': self.content['Charge'],
                    }
        else:
            raise CalculationResultError(
                "Result in `{}` seems not NormalTerminate. Please check the "
                "file manually.".format(os.path.abspath(self.path))
            )

    def get_contents(self, idx, ignoreFail=False):
        """

        Parameters
        ----------
        idx:list of int,iterable
            list of index of structure
        ignoreFail:bool
            whether to ignore fail calculation or not

        Returns
        -------

        """
        contents = []
        if not isinstance(idx, list):
            idx = list(idx)
        for item in idx:
            try:
                contents.append(self.get_content(item))
            except CalculationResultError:
                if not ignoreFail:
                    raise
                else:
                    pass
        return contents

    def get_ase_atoms(self, idx):
        from ase import Atoms
        AtomsList = []
        for item in idx:
            stuc = self.get_content(item)
            symbols, positions = structureDictParser(stucDict=stuc["StructureDict"])
            at = Atoms(
                numbers=symbols,
                positions=positions,
                info={"SpinMulti": stuc["SpinMulti"],
                      "Charge": stuc["Charge"]}
            )
            AtomsList.append(at)
        return AtomsList

    def get_energy(self, idx):
        # Use eV as energy unit
        return self.content['EnergyDict'][idx] * Hartree / eV

    def get_energies(self, idx):
        energies = []
        for item in idx:
            energies.append(self.content['EnergyDict'][item])
        return energies

    def get_all_pairs(self):
        r"""
        Get all Atoms:properties pairs

        Returns
        -------
            (ase.Atoms, list)

        """
        n = len(self.content['EnergyDict'])
        ats, en = self.get_ase_atoms(range(n)), self.get_energies(range(n))
        if len(en) >= 2:
            # for opt-freq task
            if en[-1] == en[-2]:
                ats = ats[:-1]
                en = en[:-1]
        return ats, en

    def get_final_pairs(self):
        from ase.io.gaussian import read_gaussian_out
        return ([read_gaussian_out(open(self.path, "r"))],
                [read_gaussian_out(open(self.path, "r")).get_potential_energy()]
                )


def structureDictParser(stucDict):
    symbols = []
    positions = []
    for atom in stucDict.split("\n"):
        col = atom.split()
        symbols.append(int(col[1]))
        positions.append(list(float(i) for i in col[3:]))
    return symbols, positions


if __name__ == '__main__':
    file = G16LogFiles("D:\CODE\PycharmProjects\lightMolNet\src\lightMolNet\data\DOUBLEC20_Ih_1.log")
    print(file.get_final_pairs())
