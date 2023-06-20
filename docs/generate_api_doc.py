# This script goes through the list of classes that should appear in the API Reference
# and creates the corresponding structure in the docs/api directory
# When the API is stable, we may want to remove this script to allow manually tuning the presentation of the differents elements

import os

classes_to_document = """
model.Type
model.Object
model.Fluent
model.Problem
model.InstantaneousAction
model.DurativeAction
model.htn.HierarchicalProblem
model.htn.Task
model.htn.Method
plans.Plan
plans.SequentialPlan
plans.PartialOrderPlan
plans.STNPlan
"""

base_dir = "api/"


def _write_to_file(filename: str, content):
    with open(base_dir + filename, "w") as f:
        f.write(content)


def _add_to_index(dir: str, name):
    f = open(base_dir + dir + "index.rst", "a")
    f.write(f"   {name}\n")
    f.close()


template = """
{classname}
=============================

.. autoclass:: unified_planning.{full_classname}
   :members:
   :inherited-members:
"""

index_template = """
{modulename}
=============================

.. toctree::
   :maxdepth: 1

"""


def generate():
    created = set()

    classes = classes_to_document.splitlines()
    for clazz in classes:
        clazz = clazz.strip()
        if clazz == "":
            continue
        parts = clazz.split(".")
        path = parts[:-1]
        name = parts[-1]
        full_path = ""
        prev = None
        for dir in path:
            full_path += dir + "/"
            os.makedirs(base_dir + full_path, exist_ok=True)
            if full_path not in created:
                print(f"Creating index: {full_path}")
                created.add(full_path)
                content = index_template.format(modulename=dir)
                _write_to_file(full_path + "index.rst", content)
                if prev:
                    _add_to_index(prev, f"{dir}/index")
            prev = full_path
        _add_to_index(prev, name)
        content = template.format(classname=name, full_classname=clazz)
        filename = "/".join(parts) + ".rst"
        _write_to_file(filename, content)
