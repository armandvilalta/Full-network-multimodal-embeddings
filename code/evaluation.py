"""
Evaluation code for multimodal ranking
Throughout, we assume 5 captions per image, and that
captions[5i:5i+5] are GT descriptions of images[i]
"""
import numpy
from collections import OrderedDict

from datasets import load_dataset
from datasource import Datasource
import tools

def ranking_eval_Nfold(model, n_fold=1, subset='val'):
    """
    Evaluate a trained model on either val or test of the dataset it was trained on
    Evaluate separately on n_fold image splits, and average the metrics
    Parameters:
    -----------
    model: dict
        Dictionay containing the parameters of the current model
    n_fold: int
        Number of image splits to be evaluated on.
        Only supported n_fold=1 with provided datasets.
    subset: str
        subset to perform the evaluation on.
        One of: 'val', 'test'

    Returns:
    --------
    results_dict: dict
        Dictionary containing the evaluaton results.
        Structured as results_dict['cap_ret', 'img_ret']['r1', 'r5', 'r10', 'medr'] 
    score: float
        Score obtained, the sum of recalls for both problems caption retrival and image retrieval.
    """

    results = []

    for fold in range(n_fold):
        print 'Loading fold ' + str(fold)
        dataset = load_dataset(dataset_name=model['options']['data'],
                       embedding= model['options']['embedding'],
                       path_to_data = model['options']['data_path'],
                       test_subset=model['options']['test_subset'],
                       load_train=False,
                       fold=fold)
        caps, ims = Datasource(dataset[subset], model['worddict']).all()

        print 'Computing results...'
        c_emb = tools.encode_sentences(model, caps)
        i_emb = tools.encode_images(model, ims)

        errs = tools.compute_errors(model, c_emb, i_emb)


        r = t2i(errs)
        print "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(r)

        ri = i2t(errs)
        print "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(ri)
        results.append(r + ri)

    print("-----------------------------------")
    print("Mean metrics: ")
    mean_metrics = numpy.array(results).mean(axis=0).flatten()
    print "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(mean_metrics[:5])
    print "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % tuple(mean_metrics[5:])

    # Join everything in a dict
    results_dict = OrderedDict([('cap_ret',OrderedDict([])), ('img_ret',OrderedDict([]))])
    # Caption retrieval (image to text)
    results_dict["cap_ret"]["r1"]   = mean_metrics[5]
    results_dict["cap_ret"]["r5"]   = mean_metrics[6]
    results_dict["cap_ret"]["r10"]  = mean_metrics[7]
    results_dict["cap_ret"]["medr"] = mean_metrics[8]
    # Image retrieval (text to image)
    results_dict["img_ret"]["r1"]   = mean_metrics[0]
    results_dict["img_ret"]["r5"]   = mean_metrics[1]
    results_dict["img_ret"]["r10"]  = mean_metrics[2]
    results_dict["img_ret"]["medr"] = mean_metrics[3]
    score = mean_metrics[0:3].sum() + mean_metrics[5:8].sum()
    return results_dict, score

def build_results_dict(r1, r5, r10, medr, r1i, r5i, r10i, medri):
    """
    Join results obtained in a dict
    """
    
    results_dict = OrderedDict([('cap_ret',OrderedDict([])), ('img_ret',OrderedDict([]))])
    # Caption retrieval (image to text)
    results_dict["cap_ret"]["r1"] = r1
    results_dict["cap_ret"]["r5"] = r5
    results_dict["cap_ret"]["r10"] = r10
    results_dict["cap_ret"]["medr"] = medr
    # Image retrieval (text to image)
    results_dict["img_ret"]["r1"] = r1i
    results_dict["img_ret"]["r5"] = r5i
    results_dict["img_ret"]["r10"] = r10i
    results_dict["img_ret"]["medr"] = medri
    return results_dict

def t2i(c2i):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    """

    ranks = numpy.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = numpy.argsort(d_i)

        rank = numpy.where(inds == i/5)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])
   

def i2t(c2i):
    """
    Images -> Text (Image Annotation)
    c2i: (5N, N) matrix of caption to image errors
    """

    ranks = numpy.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = numpy.argsort(d_i)

        rank = numpy.where(inds//5 == i)[0][0] # Here inds is the caption index and i the image index
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])
