# Restores packaging._structures for Isaac Sim's prebundled torch.
# The prebundle vendors packaging>=22 which dropped this file,
# but torch._vendor.packaging.version still imports it.


class InfinityType:
    def __repr__(self):
        return "Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return isinstance(other, self.__class__)

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __neg__(self):
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self):
        return "-Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return isinstance(other, self.__class__)

    def __neg__(self):
        return Infinity


NegativeInfinity = NegativeInfinityType()
