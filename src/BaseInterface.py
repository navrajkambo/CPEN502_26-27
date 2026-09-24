from abc import ABC, abstractmethod


class BaseInterface(ABC):
    @abstractmethod
    def __init__(self): pass

    # @abstractmethod
    # def Load(self): pass

    # @abstractmethod
    # def Save(self): pass

    # @abstractmethod
    # def Train(self): pass