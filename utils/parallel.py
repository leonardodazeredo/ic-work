from tqdm import tqdm
import multiprocessing


def map_in_parallel(function, values_list, with_progress_bar=False):
    '''
    Apply the 'function' to every value of 'value list' and return a list of
    equal length with the results.
    '''
    results = list()
    pool = multiprocessing.Pool()

    if with_progress_bar:
        for r in tqdm(pool.imap_unordered(function, values_list), total=len(values_list)):
            results.append(r)
    else:
        results = pool.imap_unordered(function, values_list)

    pool.close()
    pool.join()
    return results
