#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:38:19 2018

@author: ville
"""

from abc import ABC, abstractmethod


class BaseExtractor(ABC):

    def __init__(self):
        self.fs = 8000

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def extract(self, audio):
        pass
