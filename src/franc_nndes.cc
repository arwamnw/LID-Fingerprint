/* franc_nndes.cc (main) */

/* To test edge precision of true KNN graph and NN-Descent estimated KNN graph
 * The number of neighborhoods to test (k) can be smaller than K (for NN-Descent)
 *
 * Jichao Sun (js87@njit.edu)
 *
 * July 30, 2013
 * July 29, 2013
 */

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <cassert>
using namespace std;

#include "DenMatSin.h"
#include "global.h"
using namespace kprop;

#include "nndes/nndes.h"
#include "nndes/nndes-data.h"
#include "franc.h"
using namespace nndes;

#include "franc.h"

void printUsage(void)
{
    printf("usage: franc_nndes  [--verbose]\n");
    printf("                    --pwd=<path-to-working-directory>\n");
    printf("                    --ped=<path-to-experiment-directory>\n");
    printf("                    --dataset=<dataset-name>\n");
    printf("       The following parameters are for NN-Descent:\n");
    printf("                    --pre=<max #iterations-for-initial-NN-Descent>\n");
    printf("                    --K=<#KNN-for-NN-Descent>\n");
    printf("                    [--rho=<sample-rate-of-NN-Descent> default 1.0]\n");
    printf("                    [--Delta=<tolerance-value-of-NN-Descent> default 0.001]\n");
    printf("       The following parameters are for test:\n");
    printf("                    [--k=<#KNN-for-test/propagation> default K]\n");
}


int main(int argc, char **argv)
{
    /* read parameters */
    /* system parameters */
      bool verbose = false;
    string pwd;            /* directory holding the dataset */
    string ped;            /* directory holding other information of the dataset */
    string dataset;        /* name of the dataset */
    /* NN-Descent */
       int pre = -1;       /* max # of iters for initial NN-Descent */
       int K = -1;         /* #KNN for NN-Descent */
     float rho = 1.0F;     /* sample rate of NN-Descent */
     float Delta = 0.001F; /* tolerance value of NN-Descent */
    /* test */
       int small_k = -1;   /* final k used for propagation or edge precision
                              test */

    struct option long_options[] =
        {
            /* system parameters */
            {"verbose",   no_argument,       0,    'v'},
            {"pwd",       required_argument, 0,    'w'}, /* w: working */
            {"ped",       required_argument, 0,    'e'}, /* e: experiment */
            {"dataset",   required_argument, 0,    'n'}, /* n: dataset name */
            /* NN-Descent */
            {"pre",       required_argument, 0,    'p'},
            {"K",         required_argument, 0,    'K'},
            {"rho",       required_argument, 0,    'S'}, /* S: same as DCL */
            {"Delta",     required_argument, 0,    'D'},
            /* test */
            {"k",         required_argument, 0,    'k'},
            {0,           0,                 0,    0}
        };
    int option_index = 0;
    int c;
    while ( (c = getopt_long(argc, argv,
                    "vw:e:n:p:K:S:D:k:",
                    long_options, &option_index)) != -1)
    {
        switch(c)
        {
            case 'v':
                verbose = true;
                break;
            case 'w':
                pwd = string(optarg);
                break;
            case 'e':
                ped = string(optarg);
                break;
            case 'n':
                dataset = string(optarg);
                break;
            case 'p':
                pre = string_as_T<int>(string(optarg));
                break;
            case 'K':
                K = string_as_T<int>(string(optarg));
                break;
            case 'S':
                rho = string_as_T<float>(string(optarg));
                break;
            case 'D':
                Delta = string_as_T<float>(string(optarg));
                break;
            case 'k':
                small_k = string_as_T<int>(string(optarg));
                break;
            case '?':
                printUsage();
                exit(EXIT_FAILURE);
                break;
            default:
                printUsage();
                exit(EXIT_FAILURE);
                break;
        }
    }
    if (optind < argc && verbose)
    {
        KPROP_WARN("non-option argument(s) ignored: ");
        while (optind < argc)
        {
            fprintf(stderr, "%s ", argv[optind++]);
        }
        printf("\n");
    }
    /* check configurations */
    if (pwd == string(""))
    {
        KPROP_ERROR("invalid path to working directory.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (pwd == string(""))
    {
        KPROP_ERROR("invalid path to experiment directory.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (dataset == string(""))
    {
        KPROP_ERROR("invalid dataset name.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (pre < 1)
    {
        KPROP_ERROR("invalid max number of iterations for initial NN-Descent.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (K < 1)
    {
        KPROP_ERROR("invalid K for NN-Descent.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (rho > 1.0F || rho <= 0.0F)
    {
        KPROP_ERROR("invalid sampling rate for NN-Descent.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (Delta <= 0.0F)
    {
        KPROP_ERROR("invalid tolerance value for NN-Descent.");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (small_k < 1 || small_k > K)
    {
        KPROP_WARN("invalid k for test/propagation. Will use K's value %d.\n",
                K);
        small_k = K;
    }

    if (verbose)
    {
        printf("#                  Path of working directory: %s\n",
                                                          pwd.c_str());
        printf("#               Path of experiment directory: %s\n",
                                                          ped.c_str());
        printf("#                               Dataset name: %s\n",
                                                          dataset.c_str());
        printf("# Max # of iterations for initial NN-Descent: %d\n", pre);
        printf("#                            K of NN-Descent: %d\n", K);
        printf("#                Sampling rate of NN-Descent: %f\n", rho);
        printf("#              Tolerance value of NN-Descent: %f\n", Delta);
        printf("#                           k of propagation: %d\n", small_k);
        printf("\n");
    }

    /* loading descriptors in KProp dvf format */
    if (verbose)
    {
        printf("Loading descriptors...\n");
    }

    string p_data_dvf = pwd + dataset + string(".dvf");
    DenMatSin desc(p_data_dvf, true);
    int nr_data = desc.getM();
    int dim     = desc.getN();

    /* loading true labels */
    if (verbose)
    {
        printf("Loading true labels...\n");
    }
    string p_data_lab = ped + dataset + string(".matlab.lab");
    FILE* fp = fopen(p_data_lab.c_str(), "rt");
    assert(fp != NULL);
    int * true_lab = new int[nr_data];
    for (int n = 0; n < nr_data; ++n)
    {
        int c = fscanf(fp, "%d\n", true_lab+n);
        assert(c == 1);
    }
    fclose(fp);

    /* loading ture kNN */
    if (verbose)
    {
        printf("Loading true KNNs...\n");
    }
    string p_data_knn_1 = ped + dataset + string(".knn.1");
    DenMatSin trueKNNs(p_data_knn_1, true);
    assert(trueKNNs.getM() == nr_data);
    assert(trueKNNs.getN() > K);

    /* to test original edge precision (using true KNN graph) */
    printf("Computing edge precision of true KNN graph... ");
    printEdgePrecision(trueKNNs, true_lab, nr_data, small_k);
    fflush(stdout);

    /* NN-Descent */
    if (verbose)
    {
        printf("Performing NN-Descent...\n");
    }

    Dataset<float> desc_for_nndes; /* feature values stored here will change */
    desc_for_nndes.load(&desc);
    OracleL2<Dataset<float> > oracle_for_nndes(desc_for_nndes);
    assert(nr_data == desc_for_nndes.size());
    assert(dim == desc_for_nndes.getDim());
    NNDescent<OracleL2<Dataset<float> > > 
        nndes(nr_data, K, rho, oracle_for_nndes, GRAPH_BOTH);
    float total = float(nr_data) * (nr_data - 1) / 2; /* brutal force
                                                         cost */
    for (int it = 0; it < pre; ++it)
    {
        int t = nndes.iterate(); /* do an NN-Descent iteration 
                                    t is actually an estimation of number
                                    of updated items 
                                    (bug 2 in the email, not fixed, not crucial)
                                  */


        float update_rate = (float) t / (K * nr_data); /* update rate 
                                                        * although an estimation,
                                                        * will lead to converge in
                                                        * practice
                                                        */
        float cost_rate = (float) nndes.getCost() / total;

        float recall = 0.0F;
        const vector<KNN> &tempKNNs = nndes.getNN();
        for (int n = 0; n < nr_data; ++n)
        {
            recall += nndes::recall( &(trueKNNs(n, 1)), tempKNNs[n], K);
        }
        recall /= (float) nr_data;

        if (verbose)
        {
            printf("Iter %d, UpdateRate %.2f%%, CostRate %.2f%%, "
                    "Recall %.2f%%\n",
                    it+1, update_rate*100.0F, cost_rate*100.0F, recall*100.0F);
        }
        if (update_rate < Delta) break;
    }
    
    const vector<KNN> & estKNNs = nndes.getNN();
    /* to test edge precision (using NN-Descent produced KNN graph) */
    printf("Computing edge precision of KNN graph by NN-Descent... ");
    printEdgePrecision(estKNNs, true_lab, nr_data, small_k);
    fflush(stdout);

    delete[] true_lab;
    return 0;
}
