from keras.api._v2.keras import preprocessing


 
#------------------------------------------------------------------------------
def sequence_padding(sequences, pad_length, output = 'array'):

    '''
    sequence_padding(sequences, pad_value, pad_length, output='array')

    Pads a list of sequences to a specified length with a specified value.

    Keyword arguments:
        sequences (list): A list of sequences to be padded.
        pad_value (int): The value to use for padding.
        pad_length (int): The length to pad the sequences to.
        output (str): The format of the output. If 'array', the function returns a list of 
        padded sequences as numpy arrays. If 'string', the function returns a list of padded sequences as strings.

    Returns:
        padded_text (list): A list of padded sequences in the specified output format.
    
    '''
    padded_text = preprocessing.sequence.pad_sequences(sequences, maxlen=pad_length, value=0, 
                                                        dtype='int32', padding='post')
    if output == 'string':
        padded_text_str = []
        for x in padded_text:
            x_string = ' '.join(str(i) for i in x)
            padded_text_str.append(x_string)
        padded_text = padded_text_str          
    
    return padded_text   

   
