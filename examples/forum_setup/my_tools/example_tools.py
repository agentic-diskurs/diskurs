

def my_func1(name: str) -> str:
    """
    Returns a greeting message with the provided name.

    :param name: The name to greet.
    :return: The greeting message.
    """
    return f"Hello, {name}!"

def create_my_func2(configs):
    foo = configs["foo"]
    baz = configs["baz"]
    def my_func2(bar: str) -> str:
        """
        Returns a farewell message with the provided name.

        :param bar: The name to bid farewell.
        :return: The farewell message.
        """
        return f"Goodbye, {bar}, remember {foo} and {baz}!"
    return my_func2