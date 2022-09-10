def check_command_line_arguments(in_arg):
    """
    For Lab: Classifying Images - 7. Command Line Arguments
    Prints each of the command line arguments passed in as parameter in_arg, 
    assumes you defined all three command line arguments as outlined in 
    '7. Command Line Arguments'
    Parameters:
     in_arg -data structure that stores the command line arguments object
    Returns:
     Nothing - just prints to console  
    """
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     dir =", in_arg.dir, 
              "\n     arch =", in_arg.arch, "\n     learning_rate =", in_arg.learning_rate,
              "\n     epochs =", in_arg.epochs,  "\n     hidden_units =", in_arg.hidden_units,
              "\n     device =", in_arg.device)
