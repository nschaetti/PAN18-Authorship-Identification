# -*- coding: utf-8 -*-
#

# Imports
import json
import os


# Write answer JSON file
def write_answer(answers_path, problem, answers):
    """
    Write answer JSON file
    :param answers_path:
    :param problem:
    :param answers:
    :return:
    """
    # Problem file name
    problem_filename = u"answers-problem%05d.json" % (problem)

    # File path
    problem_file_path = os.path.join(answers_path, problem_filename)

    # Open
    json.dump(answers, open(problem_file_path, 'wb'))
# end write_answer
