import copy
from typing import Dict, Hashable, Iterable, List, Sequence, TypeVar


T_Hashable = TypeVar("T_Hashable", Hashable, str, int)


def merge(old: dict, new: dict) -> dict:
    result = copy.deepcopy(old)
    stack: list = [(result, new)]
    while stack:
        old, new = stack.pop()
        for k in new:
            if k in old and isinstance(old[k], dict) and isinstance(new[k], dict):
                stack.append((old[k], new[k]))
            else:
                old[k] = new[k]
    return result


def unique(*seqs: Iterable[T_Hashable]) -> List[T_Hashable]:
    return list({el for seq in seqs for el in seq})


def iterative_topological_sort(
    graph: Dict[T_Hashable, Sequence[T_Hashable]]
) -> List[T_Hashable]:
    """
    Stolen from https://stackoverflow.com/a/47234034/15632334 with minor modifications
    after giving up on implementing it myself.

    Example:
        >>> G1 = {
        ...   "a": ["b", "c"],
        ...   "b": ["d"],
        ...   "c": ["d"],
        ...   "d": []
        ... }
        >>> iterative_topological_sort(G1)
        ['d', 'c', 'b', 'a']
        >>> G1 = {
        ...     "a": ["b", "c"],
        ...     "b": [],
        ...     "c": ["b"]
        ... }  # top. sort: "b", "c", "a"
        >>> iterative_topological_sort(G2)
        ['b', 'c', 'a']
    """
    seen = set()
    stack = []
    order = []
    q = list(graph)
    while q:
        node = q.pop()
        if node not in seen:
            seen.add(node)  # no need to append to path any more
            q.extend(graph[node])
            while stack and node not in graph[stack[-1]]:  # new stuff here!
                order.append(stack.pop())
            stack.append(node)
    order.extend(stack)
    return order


__all__ = ["merge", "unique", "iterative_topological_sort"]
