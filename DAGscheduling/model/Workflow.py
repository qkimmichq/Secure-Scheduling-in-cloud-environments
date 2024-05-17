from DAGscheduling.model.Task import Task
import random
import copy

class Workflow:
    def __init__(self, id, head_task):
        self.id = id
        self.head_task = head_task
        self.tasks = self.get_all_tasks()
        self.id_to_task = self.get_id_to_task_map()

    def get_all_tasks(self):
        """
        Get all unique tasks in sorted order
        """

        def add_tasks(tasks, task):
            tasks.update(task.children)
            for child in task.children:
                add_tasks(tasks, child)

        tasks_set = set()
        if self.head_task is None:
            return tasks_set
        else:
            add_tasks(tasks_set, self.head_task)
        return tasks_set

    def get_id_to_task_map(self):
        id_map = dict()
        for t in self.tasks:
            if not t.is_head:
                id_map[t.id] = t
        return id_map

    def __str__(self):
        return str(" workflow_id: {}, head_task: {}, tasks: {}, id_to_task: {}".format(
            self.id, self.head_task, self.tasks, self.id_to_task
        ))


