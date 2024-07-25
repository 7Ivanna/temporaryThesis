sample_modes=[1,3,5,2,4]
top=
def ordered_exit(top, bottom_array):
    totaltimebins=totaltimebins+1
    lossbins=totaltimebins
    top_temp = top[0]
    bottom_temp = bottom_array[-1]
    top[0] = bottom_temp  # Assign the last value to the first index
    bottom_array[-1] = top_temp  # Store the last value of top
    last_value_top = top[-1]  # Store the last value of top
    top[1:] = top[:-1]  # Shift elements up by one position
    top[0] = last_value_top  # Assign the last value to the first index
    last_value_bottom = bottom_array[-1]  # Store the last value of top
    bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
    bottom_array[0] = last_value_bottom  # Assign the last value to the first index 
    return 
