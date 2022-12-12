---
layout: post
title: The Curious Case of the Python Debugger 
subtitle: What happens when you try to run dictionary comprehension in the Python debugger (Pdb)? 
tags: [Python]
image: /img/pdb1.jpg
---

---
This is primarily a note to remind myself of this bug the next time I encounter it. This issue is seen
on Python version 3.10.6.

I'm a big fan of using the Python and Ipython debuggers, pdb and ipdb respectively, when trying to 
debug code. Recently, I was debugging a function. I dropped in a `import ipdb;ipdb.set_trace()` right
before a line of code that did dictionary comprehension. For some reason, I ran the dictionary comprehension
directly in the python debugger. Surprisingly, a `NameError` came back for one of the variables used
in the dictionary comprehension. This was really odd considering in the same debugger session I could 
print the variable that came back with the `NameError`. The variable also appeared in `locals()`.

You can reproduce this issue with the following code. This code defines a function. Within the function,
we define a dictionary (`d1`) and a list of the keys (`fltrlist`). The dictionary comprehension allows 
us to filter out keys which do not appear in the list `fltrlist`.

For right now ignore the list (`klist`) we created outside of the function scope.

```python
def somefunc():
    fltrlist = ["A", "B"]
    d1 = {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    }

    import pdb;pdb.set_trace()
    d2 = {
        k: v for k, v in d1.items() if k in fltrlist
    }

klist = ["A", "B"]
somefunc()
```

If you run this piece of code, enter into the debugger, and try to execute the dictionary comprehension
you will get a `NameError` for `fltrlist`. 

```bash
➜ python3 scope_test.py 
> scope_test.py(10)somefunc()
-> d2 = {
(Pdb) {k: v for k, v in d1.items() if k in fltrlist}
*** NameError: name 'fltrlist' is not defined
(Pdb) print(fltrlist)
['A', 'B']
```

Let us return to the variable `klist`. This variable is defined in the global scope. Interestingly, if we 
reference this variable when executing the dictionary comprehension it works.

```bash
➜ python3 scope_test.py
> scope_test.py(11)somefunc()
-> d2 = {
(Pdb) {k: v for k, v in d1.items() if k in klist}
{'A': [1, 2, 3], 'B': [4, 5, 6]
```

So a variable that was in the local scope but not in the global scope returned a `NameError` while a
variable in the global scope worked just fine.

Let's dig a little deeper into pdb. In the [back-end](https://github.com/python/cpython/blob/d91de288e73c67805e4c838b5f770ab7ec3661f9/Lib/pdb.py#L459), pdb uses Python's `exec()` and `compile()` to 
execute commands typed into the prompt. We can try these directly with this example.

```python
def somefunc():

    fltrlist = ["A", "B"]
    d1 = {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    }

    exec(compile('{k: v for k, v in d1.items() if k in fltrlist}',
                 'string', 'exec'), globals(), locals())


klist = ["A", "B"]
somefunc()
```

When we run this we see the same issue.

```bash
➜ python3 scope_test.py
Traceback (most recent call last):
  File "/home/ebaumer/code/temp/scope_test.py", line 17, in <module>
    somefunc()
  File "/home/ebaumer/code/temp/scope_test.py", line 10, in somefunc
    exec(compile('{k: v for k, v in d1.items() if k in fltrlist}',
  File "string", line 1, in <module>
  File "string", line 1, in <dictcomp>
NameError: name 'fltrlist' is not defined
```

If we try with the globally scoped variable `klist`, it works!

```python
def somefunc():

    fltrlist = ["A", "B"]
    d1 = {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    }

    exec(compile('{k: v for k, v in d1.items() if k in klist}',
                 'string', 'exec'), globals(), locals())


klist = ["A", "B"]
somefunc()
```

We can actually go further down this rabbit hole and use Python's disassembler for bytecode
to see what is going on with the dictionary comprehension.

```python
import dis

def somefunc():

    fltrlist = ["A", "B"]
    d1 = {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    }

    dis.dis(compile('{k: v for k, v in d1.items() if k in fltrlist}',
                 'string', 'exec'))


klist = ["A", "B"]
somefunc()
```

The output of this reveals something very interesting.

```bash
➜ python3 scope_test.py
  1           0 LOAD_CONST               0 (<code object <dictcomp> at 0x7f8aa02fa130, file "string", line 1>)
              2 LOAD_CONST               1 ('<dictcomp>')
              4 MAKE_FUNCTION            0
              6 LOAD_NAME                0 (d1)
              8 LOAD_METHOD              1 (items)
             10 CALL_METHOD              0
             12 GET_ITER
             14 CALL_FUNCTION            1
             16 POP_TOP
             18 LOAD_CONST               2 (None)
             20 RETURN_VALUE

Disassembly of <code object <dictcomp> at 0x7f8aa02fa130, file "string", line 1>:
  1           0 BUILD_MAP                0
              2 LOAD_FAST                0 (.0)
        >>    4 FOR_ITER                11 (to 28)
              6 UNPACK_SEQUENCE          2
              8 STORE_FAST               1 (k)
             10 STORE_FAST               2 (v)
             12 LOAD_FAST                1 (k)
             14 LOAD_GLOBAL              0 (fltrlist)
             16 CONTAINS_OP              0
             18 POP_JUMP_IF_FALSE        2 (to 4)
             20 LOAD_FAST                1 (k)
             22 LOAD_FAST                2 (v)
             24 MAP_ADD                  2
             26 JUMP_ABSOLUTE            2 (to 4)
        >>   28 RETURN_VALUE
```

The second part is the disassembly of the dictionary comprehension. You may notice that within the
dictionary comprehension, python is trying to load `fltrlist` from the global scope. The bug becomes
aparent because the variable is defined within the function scope or the local scope and is not
in the global scope.

Will you ever encounter this bug? Maybe not. But it was fun learning how the Python debugger worked.
