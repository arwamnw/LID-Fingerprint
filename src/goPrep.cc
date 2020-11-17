/* goPrep.cc (main) */

/* Prepares the dataset:
 *  1) the kNN matrix for all data items with respect to the user-suppplied
 *     distance function (the first column is the item itself), and
 *  2) the distance matrix corresponding to 1) (the first column is 0.0F)
 *  3) full distance matrix (optional, and when there is enough memory)
 *  4) distance distribution file
 *  5) accumulated distance distribution file
 *  all files in ASCII
 * Jichao Sun (js87@njit.edu)
 *
 * Sept 4, 2013
 *     report min, max, avg distances (over all possible pairs)
 *     when memory is enough
 *
 * June 19, 2013
 *     some old parameters' shortnames changed (make consistent with
 *     goProp.cc)
 *
 * June 5, 2013 Initialized based on goPrep.cc (last modified on Mar 14,
 * 2013) from kprop.6
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cfloat>   /* for FLT_MAX */
#include <ctime>    /* for random number */
#include <cassert>
#include <getopt.h> /* to parse command line arguments */
#include <string>
using std::string;
#include <exception>
using std::bad_alloc;
#include <fstream>
using std::ifstream;

#include "global.h"
#include "DenMatSin.h"
using namespace kprop;

/* sample size (#slots) of distance distributions */
#define DIST_SAMPLE_SIZE 50

typedef struct Parameters_
{
    string pwd;      /* path of working directory, needs trailing slash */
    string dataset;  /* dataset name */
    string function; /* distance function name */
    string K;        /* nearest neighbor list length */
    string ped;      /* path of experiment directory, needs trailing slash */
    bool   dist;     /* whether or not output full distance matrix */
} Parameters;

void printUsage(void)
{
    printf("usage: goPrep --pwd=<path-of-working-directory>\n");
    printf("              --dataset=<dataset-name>\n");
    printf("              --function=<L1|L2|VA>\n");
    printf("              --K=<nearest-neighbor-list-length>\n");
    printf("              --ped=<path-of-experiment-directory>\n");
    printf("              [--dist]\n");
}

int main(int argc, char** argv)
{
    Parameters par = {string(""), string(""), string(""),
                      string(""), string(""), false};
    /* parsing command line arguments */
    struct option long_options[] = 
        {
            {"pwd",       required_argument, 0, 'w'},
            {"dataset",   required_argument, 0, 'n'},
            {"function",  required_argument, 0, 'f'},
            {"K",         required_argument, 0, 'K'},
            {"ped",       required_argument, 0, 'e'}, /* Path */
            {"dist",      no_argument,       0, 'M'}, /* distance Matrix */
            {0,           0,                 0, 0}
        };
    int option_index = 0;
    int c;
    while ( (c = getopt_long( argc, argv, "w:n:f:K:e:M",
                            long_options, &option_index)) != -1 )
    {
        switch (c)
        {
            case 'w':
                par.pwd = string(optarg);
                break;
            case 'n':
                par.dataset = string(optarg);
                break;
            case 'f':
                par.function = string(optarg);
                break;
            case 'K':
                par.K = string(optarg);
                break;
            case 'e':
                par.ped = string(optarg);
                break;
            case 'M':
                par.dist = true;
                break;
            case '?':
                /* error message has already been printed */
                printUsage();
                exit(EXIT_FAILURE);
                break;
            default:
                /* should not be here */
                printUsage();
                exit(EXIT_FAILURE);
                break;
        }
    }
    if (optind < argc)
    {
        KPROP_WARN("non-option argument(s) ignored: ");
        while (optind < argc)
        {
            fprintf(stderr, "%s ", argv[optind++]);
        }
        printf("\n");
    }

    /* validate parameters */
    /* parameters required */
    if (par.pwd == string("") || par.dataset == string("")
            || par.function == string("") || par.K == string("")
            || par.ped == string(""))
    {
        KPROP_ERROR("mandatory parameter(s) missing.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    
    /* no need to check pwd and dataset and ped */

    /* distance function */
    float (*dist_func) (const float *, const float *, int);
    if (par.function == string("L1"))
    {
        dist_func = l1Dist;
    }
    else if (par.function == string("L2"))
    {
        dist_func = l2Dist;
    }
    else if (par.function == string("VA"))
    {
        dist_func = vaDist;
    }
    else
    {
        KPROP_ERROR("unknown distance function '%s'.\n",
                    par.function.c_str());
        printUsage();
        exit(EXIT_FAILURE);
    }

    /* K --- part 1 */
    int K = string_as_T<int> (par.K);
    if (K < 1)
    {
        KPROP_ERROR("K should be at least 1.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }

    printf("   Path of working directory: %s\n", par.pwd.c_str());
    printf("                Dataset name: %s\n", par.dataset.c_str());
    printf("           Distance function: %s\n", par.function.c_str());
    printf("          Length of kNN list: %d\n", K);
    printf("Path of experiment directory: %s\n", par.ped.c_str());
    printf("     Output distance matrix?: %s\n", par.dist ? "yes": "no");

    /* read descriptors */
    string p_data_dvf = par.pwd + par.dataset + string(".dvf");
    printf("> Reading descriptors...\n");
    fflush(stdout);
    DenMatSin* descs = new DenMatSin(p_data_dvf, true);
    int nr_data = descs->getM();
    int dim     = descs->getN();
    printf("  Number of data items: %d\n", nr_data);
    printf("  Descriptor dimension: %d\n", dim);
    if (nr_data < 2)
    {
        KPROP_ERROR("at least 2 data items required.\n");
        exit(EXIT_FAILURE);
    }
    if (dim < 1)
    {
        KPROP_ERROR("invalid descriptor dimension.\n");
        exit(EXIT_FAILURE);
    }
    /* K --- part 2 */
    if (K > nr_data - 1)
    {
        KPROP_ERROR("K should be at most %d.\n", nr_data - 1);
        printUsage();
        exit(EXIT_FAILURE);
    }
    /* read labels */
    string p_data_lab = par.pwd + par.dataset + string(".lab");
    printf("> Reading ground truth labels... ");
    fflush(stdout);
    ifstream ifs_data_lab(p_data_lab.c_str());
    if (ifs_data_lab == NULL)
    {
        KPROP_ERROR("could not open ground truth label file '%s' to read.\n",
                    p_data_lab.c_str());
        exit(EXIT_FAILURE);
    }
    string line;
    if (!getline(ifs_data_lab, line))
    {
        KPROP_ERROR("corrupted header of ground truth label file '%s'.\n",
                    p_data_lab.c_str());
        exit(EXIT_FAILURE);
    }
    trim(line);
    if (nr_data != string_as_T<int>(line))
    {
        KPROP_ERROR("inconsistent number of data items defined in "
                    "ground truth label file '%s'.\n", p_data_lab.c_str());
        exit(EXIT_FAILURE);
    }
    vector<string>* labels = new vector<string>;
    labels->reserve(nr_data); /* no need for efficiency */
    for (int i = 0; i < nr_data; i++)
    {
        if (!getline(ifs_data_lab, line))
        {
            KPROP_ERROR("corrupted ground truth label file '%s'.\n",
                        p_data_lab.c_str());
            exit(EXIT_FAILURE);
        }
        trim(line);
        labels->push_back(line);
    }
    ifs_data_lab.close();
    printf("done.\n");
    fflush(stdout);

    /* compute knn matrices & dd */
    printf("> Computing kNN matrices and distance distributions...\n");
    fflush(stdout);
    string p_data_knn_1 = par.ped + par.dataset + string(".knn.1");
    string p_data_knn_2 = par.ped + par.dataset + string(".knn.2");
    string p_dd_1       = par.ped + par.dataset + string(".dd.1");
    string p_dd_2       = par.ped + par.dataset + string(".dd.2");
    /* optional distance matrix */
    string p_data_dist  = par.ped + par.dataset + string(".dist");
    FILE* fp = NULL;
    /* kNN matrices (the first column saves the item itself with distance
     * 0 */
    DenMatSin *knn_1 = new DenMatSin(nr_data, K+1);
    DenMatSin *knn_2 = new DenMatSin(nr_data, K+1);
    ifpair    *sort  = new ifpair[nr_data];

    float  max_dist  = 0.0F;
    float  min_dist  = FLT_MAX; 
    float  avg_dist  = 0.0F;
    int    nr_inter  = 0;
    int    nr_intra  = 0;
    float  sum_inter = 0.0F;
    float  sum_intra = 0.0F;
    int    inter_dist_count[DIST_SAMPLE_SIZE];
    int    intra_dist_count[DIST_SAMPLE_SIZE];

    /* try to compute the whole distance matrix (the upper triangular
     * matrix (without diagonal)
     * if failed (no memory) compute the distances for each data item
     * at a time
     */
    float *dists         = NULL;
    bool  *is_intra_dist = NULL;
    printf("  Trying to compute the whole distance matrix... ");
    fflush(stdout);
    bool compute_all     = false; /* a flag */
    int  nr_samples;
    int  nr_dists;
    try
    {
        nr_samples    = nr_data;
        nr_dists      = nr_samples * (nr_samples-1)/2;
        dists         = new float[nr_dists];
        is_intra_dist = new bool [nr_dists];
        /* successfully allocated */
        compute_all   = true;
        printf("memory allocated.\n");
        fflush(stdout);
        if (par.dist)
        {
            fp = fopen(p_data_dist.c_str(), "wt");
            if (fp == NULL)
            {
                KPROP_ERROR("could not open distance matrix file '%s' to "
                            "write.\n", p_data_dist.c_str());
                exit(EXIT_FAILURE);
            }
            fprintf(fp, "%d %d\n", nr_data, nr_data);
        }
    }
    catch (bad_alloc &)
    {
        nr_samples    = KPROP_MIN(10000, nr_data);
        nr_dists      = nr_samples * (nr_samples-1)/2;
        if (dists != NULL)
            delete[] dists;
        dists         = new float[nr_dists];
        is_intra_dist = new bool [nr_dists];
        compute_all   = false;
        printf("allocating memory failed.\n");
        if (par.dist)
        {
            KPROP_WARN("cannot output distance matrix.\n");
        }
        fflush(stdout);
    }
    for (int i = 0; i < nr_samples; ++i)
    {
        is_intra_dist[i] = false;
    }

    int nr_cand;
    int temp = nr_data / 10; /* for processing bar */
    printf("  ");
    if (compute_all)
    {
        for (int i = 0; i < nr_data; i++)
        {
            nr_cand = 0;
            for (int j = 0; j < i; j++) /* add already computed distances */
            {
                int ind = (2*nr_data-1-j)*j/2 + (i-j-1); /* index of dist */
                sort[nr_cand].index = j;
                sort[nr_cand++].weight = dists[ind];
            }
            for (int j = i+1; j < nr_data; j++) /* compute a new distance */
            {
                int ind = (2*nr_data-1-i)*i/2 + (j-i-1); /* index of dist */
                float dist
                    = (*dist_func)(&((*descs)(i,0)), &((*descs)(j,0)), dim);
                sort[nr_cand].index = j;
                sort[nr_cand++].weight = dist;
                dists[ind] = dist;
                if (dist > max_dist) max_dist = dist;
                if (dist < min_dist) min_dist = dist;
                if ((*labels)[j] == (*labels)[i])
                {
                    is_intra_dist[ind] = true;
                    nr_intra++;
                    sum_intra += dist;
                }
                else
                {
                    nr_inter++;
                    sum_inter += dist;
                }

                avg_dist += dist;
            }
            qsort(sort, nr_cand, sizeof(ifpair), cmp_ifpairs_i);
            (*knn_1)(i,0) = i;
            (*knn_2)(i,0) = 0.0F;
            for (int k = 0; k < K; k++)
            {
                (*knn_1)(i,k+1) = (float)sort[k].index;
                (*knn_2)(i,k+1) = sort[k].weight;
            }
            /* processing bar */
            if (i+1 < nr_data)
            {
                if (temp != 0 && (i+1)%temp == 0)
                {
                    printf("%d%% ", (i+1)/temp*10);
                }
            }
            else
            {
                printf("100%%\n");
            }
            fflush(stdout);
        }
        if (par.dist) /* output distance matrix */
        {
            for (int i = 0; i < nr_data; i++)
            {
                for (int j = 0; j < nr_data; j++)
                {
                    if (i == j)
                    {
                        fprintf(fp, "%f ", 0.0F);
                    }
                    else if (j > i)
                    {
                        /* index of dist */
                        int ind = (2*nr_data-1-i)*i/2 + (j-i-1);
                        fprintf(fp, "%f ", dists[ind]);
                    }
                    else /* j < i */
                    {
                        /* index of dist */
                        int ind = (2*nr_data-1-j)*j/2 + (i-j-1);
                        fprintf(fp, "%f ", dists[ind]);
                    }
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            fp = NULL;
        }
    }
    else
    {
        /* random permutation to sample at most 10000 data items to compute
         * the distance distributions
         */
        srand(time(NULL));
        i2pair *sort2 = new i2pair[nr_data];
        /* indices less than nr_samples will be used to compute distance
         * distributions
         */
        for (int i = 0; i < nr_data; i++)
        {
            sort2[i].index  = i;
            sort2[i].weight = rand();
        }
        qsort(sort2, nr_data, sizeof(i2pair), cmp_i2pairs_i);
        int sample_count = 0;
        for (int i = 0; i < nr_data; i++)
        {
            nr_cand = 0;
            for (int j = 0; j < nr_data; j++)
            {
                if (i == j) continue;
                float dist
                    = (*dist_func)(&((*descs)(i,0)), &((*descs)(j,0)), dim);
                sort[nr_cand].index = j;
                sort[nr_cand++].weight = dist;
                if ((sort2[i].index < nr_samples)
                    && (sort2[j].index < nr_samples)
                    && (j > i))
                {
                    dists[sample_count] = dist;
                    if (dist > max_dist) max_dist = dist;
                    if (dist < min_dist) min_dist = dist;
                    if ((*labels)[j] == (*labels)[i])
                    {
                        is_intra_dist[sample_count] = true;
                        nr_intra++;
                        sum_intra += dist;
                    }
                    else
                    {
                        nr_inter++;
                        sum_inter += dist;
                    }
                    sample_count++;
                }
            }
            qsort(sort, nr_cand, sizeof(ifpair), cmp_ifpairs_i);
            (*knn_1)(i,0) = i;
            (*knn_2)(i,0) = 0.0F;
            for (int k = 0; k < K; k++)
            {
                (*knn_1)(i,k+1) = (float)sort[k].index;
                (*knn_2)(i,k+1) = sort[k].weight;
            }
            /* processing bar */
            if (i+1 < nr_data)
            {
                if (temp != 0 && (i+1)%temp == 0)
                {
                    printf("%d%% ", (i+1)/temp*10);
                }
            }
            else
            {
                printf("100%%\n");
            }
            fflush(stdout);
        }
        delete sort2;
    }
    float interval = max_dist / DIST_SAMPLE_SIZE;
    for (int k = 0; k < DIST_SAMPLE_SIZE; k++)
    {
        inter_dist_count[k] = 0;
        intra_dist_count[k] = 0;
    }
    for (int i = 0; i < nr_dists; i++)
    {
        int index = KPROP_MIN((int)(dists[i]/interval), DIST_SAMPLE_SIZE-1);
        if (is_intra_dist[i])
        {
            intra_dist_count[index] ++;
        }
        else
        {
            inter_dist_count[index] ++;
        }
    }

    /* save files */
    printf("> Saving kNN matrices... ");
    fflush(stdout);
    knn_1->save(p_data_knn_1, true,  0); /* integers */
    knn_2->save(p_data_knn_2, true, -1);
    printf("done.\n");
    fflush(stdout);
    printf("> Saving distance distributions... ");
    fflush(stdout);
    fp = fopen(p_dd_1.c_str(), "wt");
    if (fp == NULL)
    {
        KPROP_ERROR("could not open distance distribution file '%s' to "
                    "write.\n", p_dd_1.c_str());
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "# %s using ", par.dataset.c_str());
    if (compute_all)
        fprintf(fp, "all distances\n");
    else
        fprintf(fp, "using %d distances\n", nr_samples);
    fprintf(fp, "# Intra dist count: [0, %f), [%f, %f) ... [%f, %f]\n",
            interval, interval, 2* interval,
            (DIST_SAMPLE_SIZE-1)*interval, max_dist);
    fprintf(fp, "# Intra-dists: %d\n", nr_intra);
    for (int k = 0; k < DIST_SAMPLE_SIZE; k++)
    {
        fprintf(fp, "%f %f %d\n", k*interval,
                KPROP_MIN((float)intra_dist_count[k]/nr_intra*100.0F, 100.0F),
                intra_dist_count[k]);
    }
    fprintf(fp, "# Inter dist count: [0, %f), [%f, %f) ... [%f, %f]\n",
            interval, interval, 2* interval,
            (DIST_SAMPLE_SIZE-1)*interval, max_dist);
    fprintf(fp, "# Inter-dists: %d\n", nr_inter);
    for (int k = 0; k < DIST_SAMPLE_SIZE; k++)
    {
        fprintf(fp, "%f %f %d\n", k*interval,
                KPROP_MIN((float)inter_dist_count[k]/nr_inter*100.0F, 100.0F),
                inter_dist_count[k]);
    }
    fclose(fp);
    /* accumulated distributions */
    for (int k =1; k < DIST_SAMPLE_SIZE; k++)
    {
        intra_dist_count[k] = intra_dist_count[k] + intra_dist_count[k-1];
        inter_dist_count[k] = inter_dist_count[k] + inter_dist_count[k-1];
    }
    fp = fopen(p_dd_2.c_str(), "wt");
    if (fp == NULL)
    {
        KPROP_ERROR("could not open distance distribution file '%s' to "
                    "write.\n", p_dd_2.c_str());
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "# %s using ", par.dataset.c_str());
    if (compute_all)
        fprintf(fp, "all distances\n");
    else
        fprintf(fp, "using %d distances\n", nr_samples);
    fprintf(fp, "# Intra dist count: [0, %f), [%f, %f) ... [%f, %f]\n",
            interval, interval, 2* interval,
            (DIST_SAMPLE_SIZE-1)*interval, max_dist);
    fprintf(fp, "# Intra-dists: %d\n", nr_intra);
    for (int k = 0; k < DIST_SAMPLE_SIZE; k++)
    {
        fprintf(fp, "%f %f %d\n", k*interval,
                KPROP_MIN((float)intra_dist_count[k]/nr_intra*100.0F, 100.0F),
                intra_dist_count[k]);
    }
    fprintf(fp, "# Inter dist count: [0, %f), [%f, %f) ... [%f, %f]\n",
            interval, interval, 2* interval,
            (DIST_SAMPLE_SIZE-1)*interval, max_dist);
    fprintf(fp, "# Inter-dists: %d\n", nr_inter);
    for (int k = 0; k < DIST_SAMPLE_SIZE; k++)
    {
        fprintf(fp, "%f %f %d\n", k*interval,
                KPROP_MIN((float)inter_dist_count[k]/nr_inter*100.0F, 100.0F),
                inter_dist_count[k]);
    }
    fclose(fp);
    printf("done.\n");
    fflush(stdout);

    /* cleaning up */
    printf("> Cleaning up... ");
    fflush(stdout);
    delete[] is_intra_dist;
    delete[] dists;
    delete sort;
    delete knn_2;
    delete knn_1;
    delete labels;
    delete descs;
    printf("done.\n");


    int nr_dist = nr_data * (nr_data - 1) / 2;
    assert(nr_dist == nr_inter + nr_intra);
    printf("  Min, max and average distance: [%f, %f], %f\n",
           min_dist, max_dist, avg_dist / nr_dist);
    fflush(stdout);
    return 0;
}
