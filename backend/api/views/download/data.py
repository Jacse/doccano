from typing import Any, Dict, List
import traceback


class Record:

    def __init__(self,
                 id: int,
                 data: str,
                 label: List[Any],
                 user: str,
                 metadata: Dict[Any, Any],
                 annotation_relations: List[Any] = []
                 ):
        self.id = id
        self.data = data
        self.label = label
        self.annotation_relations = annotation_relations
        self.user = user
        self.metadata = metadata

        # print("Record called. Traceback:")
        # for line in traceback.format_stack():
        #     print(line.strip())

    def __str__(self):
        return f'{self.data}\t{self.label}'
