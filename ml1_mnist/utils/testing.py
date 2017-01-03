import nose

@nose.tools.nottest
def run_tests(current_path, test_module):
    params = [current_path, test_module.__file__, '--with-doctest']
    nose.run(argv=[''] + params)