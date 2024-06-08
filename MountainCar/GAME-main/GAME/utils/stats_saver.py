# Author: Minh Hua
# Date: 11/10/2022
# Purpose: This module contains code for a class that saves run statistics for the different mapping evolution algorithms.

from collections import namedtuple
import sys
import os
import json

# Named tuple that stores metadata about an experiment
MappingSearchExperimentInfo = namedtuple('MappingSearchExperimentInfo', [
    'src_task_name', 'target_task_name', 'search_algorithm',
    'algorithm_params'
])
# TrialStats = namedtuple('TrialStats', [
#     'trial_no', 'generation', 'unique_mappings', 'duplicate_mappings', 'comparisons',
#     'fitness_over_time', 'converged', 'iterations_before_convergence', 'final_mapping',
#     'final_mapping_fitness'
# ])
TrialStats = namedtuple('TrialStats', [
    'trial_no', 'generation', 'fitness_evaluations', 'comparisons',
    'best_fitness', 'best_fitness_ind_ID'
])

# TrialStats = namedtuple('TrialStats', [
#     'generation', 'comparisons', 'best_fitness'
# ])

class StatisticsSaver:
    """Saves statistics for search algorithms"""
    def __init__(
        self,
        experiment_info:MappingSearchExperimentInfo,
        trial:int=0,
        fitness_max:bool=True
    ) -> None:
        """
        Description:
            Initializes a statistics saver.

        Arguments:
            experiment_info: metadata about the search experiment.
            trial: the current trial number.
            fitness_max: whether we are minimizing or maximizing fitness

        Return:
            (None)
        """
        # save metadata
        self.experiment_info = experiment_info
        self.trial = trial
        self.fitness_max = fitness_max

        # main structure for data
        self.data = []

        # data structures for aux info
        self.mappings_seen_so_far = {}

        self.best_fitness_per_fnc_eval = []

    def analyze_population_and_log_stats(self, population:list, generation:int, comparisons:int, fitness_evaluations:int) -> None:
        # save the best fitness
        best_individual = None
        if self.fitness_max:
            best_fitness = -(sys.maxsize * 2 + 1)
        else:
            best_fitness = sys.maxsize * 2 + 1
        
        # process each mapping
        for mapping in population:
            mapping_ID = str(mapping.ID)
            # count the number of unique and duplicate mappings seen so far
            if mapping_ID not in self.mappings_seen_so_far.keys():
                self.mappings_seen_so_far[mapping_ID] = 1
            else:
                self.mappings_seen_so_far[mapping_ID] = self.mappings_seen_so_far[mapping_ID] + 1
            # save the best fitness and the best individual
            if self.fitness_max and mapping.fitness > best_fitness:
                best_fitness = mapping.fitness
                best_individual = mapping_ID
            elif not self.fitness_max and mapping.fitness < best_fitness:
                best_fitness = mapping.fitness
                best_individual = mapping_ID
        
        gen_data = TrialStats(self.trial, generation, fitness_evaluations, comparisons, best_fitness, best_individual)
        # save data
        self.data.append(gen_data)
    
    def log_best_fitness_by_fnc_eval(self, population:list):
        best_fitness_ind = sorted(population, key=lambda agent: agent.fitness, reverse=True)[0]
        best_fitness_ID = best_fitness_ind.ID
        best_fitness = best_fitness_ind.fitness
        self.best_fitness_per_fnc_eval.append(best_fitness)

    def log_mapping(self, mapping):
        mapping_ID = str(mapping.ID)
        # count the number of unique and duplicate mappings seen so far
        if mapping_ID not in self.mappings_seen_so_far.keys():
            self.mappings_seen_so_far[mapping_ID] = 1
        else:
            self.mappings_seen_so_far[mapping_ID] = self.mappings_seen_so_far[mapping_ID] + 1

    def export_data(self, path:str, file_name:str) -> None:
        """
        Description:
            Saves the dataframe to an external file.

        Arguments:
            path: the path to save the data.
            file_name: the name for the file.

        Return:
            (None)
        """
        # write unique and duplicate mappings
        with open(os.path.join(path, '{}_population_stats.txt'.format(file_name.split('.')[0])), 'w') as f:
            json.dump(self.mappings_seen_so_far, f)
        with open(os.path.join(path, file_name), 'w') as f:
            # sum number of comparisons
            # f.write('Total comparisons: {}\n'.format(sum(data_pt.comparisons for data_pt in self.data)))
            f.write("trial_no,generation,fitness_evaluations,comparisons,best_fitness,best_fitness_ind_ID\n")
            # write aux data
            for data_pt in self.data:
                f.write('{},{},{},{},{},{}\n'.format(*data_pt))

    def write_metadata(self, path:str, file_name:str) -> None:
        """
        Description:
            Saves the metadata to an external file.

        Arguments:
            path: the path to save the metadata.
            file_name: the name for the file.

        Return:
            (None)
        """
        with open(path + file_name, 'w') as f:
            sep = '-------------------------------------------------------------'
            f.write('Experiment Info \n' + sep + '\n')
            for k, v in self.experiment_info._asdict().items():
                f.write('{} = {} \n'.format(k, v))