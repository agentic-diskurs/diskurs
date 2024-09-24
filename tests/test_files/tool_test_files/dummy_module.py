# dummy_module.py
def simple_function(configs, dep1=None, dep2=None):
    return f"Configs: {configs}, Dependencies: {dep1}, {dep2}"


def create_sample_function(configs, dep1=None, dep2=None):
    def inner_function():
        return f"Inner Configs: {configs}, Inner Dependencies: {dep1}, {dep2}"

    return inner_function
