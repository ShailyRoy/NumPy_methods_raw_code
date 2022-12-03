# NumPy_methods_raw_code

you will practice using array processing libraries without writing explicit loops in your code. Not only will this make your code more concise, it will also make your code run faster, since these libraries have been heavily optimized for speed and are implemented largely in C++, not Python. This question requires that you have a recent version of numpy installed.
The file array tests.py contains several unit tests. Each unit test has a private method prefixed with an underscore (‘ ’), such as arith. These private methods use nested for- loops to process a numpy array or list of numpy arrays. Each of these private methods has a corresponding stub in the file array code.py.
Your task is to complete these stubs in array code.py so that they produce the same nu- merical results as the corresponding private methods in array tests.py. However, unlike array tests.py, you are not allowed to use any Python loops, including list comprehensions. In other words, the for and while keywords should not appear anywhere in your implemen- tation of array code.py, including your code comments. No other import statements are allowed either. Instead, you should use numpy array operations that implicitly do all the loops for you.
Each stub has comments with links to documentation of the relevant numpy functionality. Link comments are not repeated, but some subsequent stubs can be solved using the same functionality linked in one or more previous stubs (which is why some later stubs have no comments). You will have completed this question once you are able to run array tests.py and all test cases pass. The tests use random test data, so run them multiple times to make sure your code works in every corner case
