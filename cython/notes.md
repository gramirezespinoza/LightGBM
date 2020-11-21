# LightGBM and Cython


## TODO
- [x] How to make a shared pointer? (https://dmtn-013.lsst.io)
- [ ] Learn to use structs defined in c++
- [x] How to avoid hard-coded paths in cython files
- [x] Learn how to use a shared library `*.so` to use the compiled LightGBM
- [ ] Learn to properly use `distutils` and `setup.py`
- [ ] How to avoid hard-coded paths in `setup.py` files
- [ ] Learn to write/use makefiles

ADD PATHHHHHHH


## Notes

TODO

### Things to remember

* Build process `python setup.py build_ext --inplace`
* `setup.py` must have language defined `language=c++`
* `pxd` files contain definitions which work like C header files [doc](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html)
* `pxi` include files (?)
* `pyx` files contain implementation (source code)
* in `cdef` declarations I can use any data-type from C/C++

## Issues and Problems

### SOLVED: From Oct 12th 2020

**Solution**
The source files have to be added to files `pyx` to make the implementation
 available to cython. To add, include a directive at the beginning of the file:

* I'm having  issues calling methods in classes defined on C++ and wrapped by
 Cython. The class is `DatasetLoader` and the method is `LoadFromFile`
 * It seems the compiler  is calling

 ```
 DatasetLoader::LoadFromFile(char const *, int, int)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/gire/Desktop/Repositories/LightGBM/
```

 instead of calling `DatasetLoader::LoadFromFile(const char *, int, int)` and
  i dont  know why yet

### Code

* The code in `c_api.cpp` and `c_api.h` create an API for C language but it
 is not required for writing cython wrapper for python


## Library of Documents

* [Simple intro](https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/)
* [typedef](https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html)
* [cython tricks](https://gist.github.com/ctokheim/6c34dc1d672afca0676a)
* [How to use a shared library](https://stackoverflow.com/questions/16993927/using-cython-to-link-python-to-a-shared-library)

* [For PyCharm building](https://stackoverflow.com/questions/27305343/cython-not-recognizing-c11-commands)

* [C++ containers in Cython](http://tillahoffmann.github.io/2016/04/18/Cpp
-containers-in-cython.html)
