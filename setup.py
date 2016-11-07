from distutils.core import setup, Extension

processor_modules = [
    Extension('_processor',
              sources=['processor.cpp', 'Config.cpp', 'processor_wrap.cxx'],
              include_dirs=['C:/opencv/build/include/',
                            'C:/Users/jlb/Desktop/build/include/',
                            'C:/Users/jlb/Desktop/build/tesseract/api/',
                            'C:/Users/jlb/Desktop/build/tesseract/ccutil/',
                            'C:/Users/jlb/Desktop/build/tesseract/ccstruct',
                            'C:/Users/jlb/Desktop/build/tesseract/ccmain',
                            'C:/Users/jlb/Desktop/build/tesseract/classify/'],

              library_dirs=['C:/opencv/build/x64/vc12/lib',
                            'C:/Users/jlb/Desktop/build/lib',
                            'C:/Users/jlb/Desktop/build/lib/x64'],
              libraries=['opencv_world310', 'libtesseract304', 'liblept171'])
]

setup(
    name="processor",
    version="0.01",
    description="extension recoginze",
    ext_modules=processor_modules,
    py_modules=["processor"]
)