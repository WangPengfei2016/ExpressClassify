#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" c++扩展识别模块安装文件 """

from distutils.core import setup, Extension

processor_modules = [
    Extension('_processor',
              sources=['processor.h', 'Config.cpp', 'processor_wrap.cxx'],
              include_dirs=[''],
              library_dirs=[''],
              librarys=[''])
    ]

setup(name='processor',
      versions="0.01",
      description="recoginze phone number using opencv and tesseract",
      ext_modules=processor_modules,
      py_modules=["processor"]
     )
