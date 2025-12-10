#!/usr/bin/python3
'''Compute an allowed mask, number of threads and other vllm OMP parameters'''

import os
import re
from argparse import ArgumentParser

PROC_RE = re.compile(r"processor\s+:\s+(\d+)")
CORE_RE = re.compile(r"core id\s+:(\d+)")
PROPERTY_RE = re.compile(r"^([\w, ]+)\s*:\s*([\w, ]+)$")
INT_RE = re.compile(r"([0-9,\+,-]*)")
FLOAT_RE = re.compile(r"([0-9,-,\.]*)")

def make_number(arg):
    '''If it looks like a number, walks like a number it is a number'''
    if len(INT_RE.match(arg).group(1)) == len(arg):
        return int(arg)
    if len(FLOAT_RE.match(arg).group(1)) == len(arg):
        return float(arg)
    return arg

#pylint: disable=too-few-public-methods
class Cpu:
    '''CPU on Linux'''
    def __init__(self, section):
        self.properties = {}
        for line in section:
            try:
                digested = PROPERTY_RE.match(line)
                self.properties[digested.group(1)] = make_number(digested.group(2))
            except AttributeError:
                pass

#pylint: disable=too-few-public-methods
class Core:
    '''Physical core'''
    def __init__(self, cpu):
        self.core_id = cpu.properties["core id"]
        self.cpus = {cpu.properties["processor"]:cpu}

    def add_vcpu(self, cpu):
        '''Add a (v)cpu to the phys core'''
        self.cpus[cpu.properties["processor"]] = cpu

class System:
    '''CPU information for all cores'''
    def __init__(self, filename="/proc/cpuinfo"):

        self.cpus = []
        self.cores = {}
        section = []

        with open(filename, mode="r", encoding="ascii") as cpuinfo:
            data = cpuinfo.readlines()
        for line in data:
            if PROC_RE.match(line) is not None:
                if len(section) > 0:
                    self.cpus.append(Cpu(section))
                section = [line]
            else:
                section.append(line)
        self.cpus.append(Cpu(section))
        self.resolve_cores()
        self.apply_affinity_mask()

    def resolve_cores(self):
        '''Resolve cores from collected CPU info'''
        for candidate in self.cpus:
            try:
                self.cores[candidate.properties["core id"]].add_vcpu(candidate)
            except KeyError:
                self.cores[candidate.properties["core id"]] = Core(candidate)

    def apply_affinity_mask(self, pid=0):
        '''Apply affinity mask - prune all processors which are not allowed'''
        mask = os.sched_getaffinity(pid)
        for core in self.cores.values():
            admitted = {}
            for processor_id in core.cpus.keys():
                if processor_id in mask:
                    admitted[processor_id] = core.cpus[processor_id]
            core.cpus = admitted

    def possible_threads(self):
        '''Total threads usable with this affinity mask'''
        total = 0
        for core in self.cores.values():
            total += len(core.cpus)
        return total

    def possible_omp_threads(self):
        '''Total thread-per-core usable with this affinity mask'''
        total = 0
        for core in self.cores.values():
            if len(core.cpus) > 0:
                total += 1
        return total

def main():

    '''Gather CPU Info and return the usable number of OMP threads'''

    aparser = ArgumentParser(description=main.__doc__)

    aparser.add_argument(
        '-a', '--allthreads',
        help='return all possible threads, not OMP ones',
        action='store_true')

    args = vars(aparser.parse_args())

    system = System()

    if args.get("allthreads"):
        print(system.possible_threads())
    else:
        print(system.possible_omp_threads())

if __name__ == "__main__":
    main()
