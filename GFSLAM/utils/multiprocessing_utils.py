import copy

import torch
import torch.multiprocessing as mp


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise mp.queues.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    
    # torch.cuda.synchronize()
    return clone_obj

# def clone_obj(obj):
#     """
#     """
#     new_obj = copy.deepcopy(obj)

#     def convert(v):
#         if isinstance(v, torch.Tensor):
#             return v.detach().cpu().clone()
#         elif isinstance(v, (list, tuple)):
#             return type(v)(convert(x) for x in v)
#         elif isinstance(v, dict):
#             return {k: convert(x) for k, x in v.items()}
#         else:
#             return v

#     for attr, val in new_obj.__dict__.items():
#         if hasattr(new_obj.__class__, attr) and isinstance(
#             getattr(new_obj.__class__, attr), property
#         ):
#             continue
#         setattr(new_obj, attr, convert(val))

#     return new_obj
