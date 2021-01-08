import pickle
import os

def save2pkl(data, name, dir_output='output_files', band='L', mode='RAVC', **conf):

    ''' Save data to a pickle file. '''

    # update dict
    conf.update(dir_output=dir_output)
    ignore = ['perf_num','psf_num','vvc','lyotmask']
    for key in ignore :
        if key in conf.keys():
            del conf[key]
    os.makedirs(dir_output, exist_ok=True)
    filename = os.path.join(dir_output, '%s_%s_%s.pkl'%(name, band, mode))
    with open(filename, 'wb') as file:
        pickle.dump(conf, file)
        file.close()

    return filename