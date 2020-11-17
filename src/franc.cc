/* franc.cc (main) */

/* Main of LID-Fingerprint
 * based on the algorithm v2 as of July 1, 2013
 * see the algorithm outline in the upper level folder
 * The code is modified from NN-Descent Code
 * Arwa Wali (amw7@njit.edu)
 *
 * May 01, 2018
 */

#include <vector>
#include <algorithm> /* for random_shuffle */
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <limits>
#include <cfloat>
#include <omp.h>

#include <string>
#include <sstream>

using namespace std;
#include <boost/dynamic_bitset.hpp>

#include <boost/tokenizer.hpp>
using boost::tokenizer;

#include "DenMatSin.h"
#include "global.h"
using namespace kprop;

#include "nndes/nndes.h"
#include "nndes/nndes-data.h"
using namespace nndes;

#include "franc.h"



vector<float> score;

//#include "id.tpp"

bool compareScore(int i, int j) {
    return (score[i]<score[j]);
}



double evalID(	float* Q,
         const int n) {
    //cout << "inside evalID" <<endl;
    float w = Q[n-1];
    //cout << "w = "<<w<<endl;
    if (w == 0) return std::numeric_limits<float>::max();
    if (Q[0] == w) return std::numeric_limits<float>::max();
    double sum = 0;
    for (int i=0; i < n-1; i++) {
        //cout << "Q["<<i<<"] = "<<Q[i]<<endl;
        sum += log(Q[i] / w);
    }
    //cout << "sum = "<<sum<<endl;
    sum = sum / (n-1);
    //cout << "id = "<< (-1 / sum) <<endl;
    return -1 / sum;
}// end evalID

float  w_KNN_distance(	float* Q,
                      const int n) {
    //cout << "inside evalID" <<endl;
    float w = Q[n-1];
    //cout << "w = "<<w<<endl;
    if (w == 0) return std::numeric_limits<float>::max();
    if (Q[0] == w) return std::numeric_limits<float>::max();
    
    return w;
}// end w_KNN_distance

void printUsage(void)
{
    printf("usage: franc  [--verbose]\n");
    printf("              --pwd=<path-to-working-directory>\n");
    printf("              --ped=<path-to-experiment-directory>\n");
    printf("              --dataset=<dataset-name>\n");
    printf("       Parameters of NN-Descent/NNF-Descent:\n");
    printf("              --K=<#KNN-for-NN-Descent|NNF-Descent>\n");
    printf("              [--rho=<sample-rate-of-NNF-Descent> default 1.0]\n");
    printf("              --Iter=<max-#iterations-for-NNF-Descent>\n");
    printf("              --spa=<max-#dims-to-sparsify-each-time>\n");
    printf("              [--t=<-:linear|0:auto-t-Gaussian|+:t-Gaussian> default -1]\n");
//     printf("       To test KNN classification:\n");
//     printf("              [--ann=<path-to-annotations> default .]\n");
//     printf("              [--size=<%%labeled-items-per-class> default 1]\n");
//     printf("              [--exp=<experiment-number> default 1]\n");
    printf("\n");
    printf("              [--only]\n");

}


int main(int argc, char **argv)
{
    /* read parameters */
    /* system parameters */
      bool verbose = false;
    string pwd;            /* directory holding the dataset */
    string ped;            /* directory holding other information of the dataset */
    string dataset;        /* name of the dataset */
    /* NN-Descent / NNF-Descent */
       int K = -1;         /* #KNN for NN-Descent */
     float rho = 1.0F;     /* sample rate of NN-Descent */
       int Iter = -1;       /* max # of iters for NNF-Descent */
       int spa = -1;       /* max # of dims to sparsify each time */
     float fixed_t = -1.0F; /* default linear */

//     /* parameters of KNN classification */
//     string p_ann = "";
//        int size = 1;
//        int exp = 1;

    struct option long_options[] =
        {
            /* system parameters */
            {"verbose",   no_argument,       0,    'v'},
            {"pwd",       required_argument, 0,    'w'}, /* w: working */
            {"ped",       required_argument, 0,    'e'}, /* e: experiment */
            {"dataset",   required_argument, 0,    'n'}, /* n: dataset name */
            /* NN-Descent / NNF-Descent */
            {"K",         required_argument, 0,    'K'},
            {"rho",       required_argument, 0,    'S'}, /* S: same as DCL */
            {"Iter",      required_argument, 0,    'I'},
            {"spa",       required_argument, 0,    'z'}, /* z: zero */
            {"t",         required_argument, 0,    't'},
//             /* KNN classification */
//             {"ann",       required_argument, 0,    'a'},
//             {"size",      required_argument, 0,    's'},
//             {"exp",       required_argument, 0,    'x'},

            {"only",      no_argument,       0,    'o'},
            {0,           0,                 0,    0}
        };

    int option_index = 0;
    int c;
    int    simtype; /* similarity type: -1 for linear,
                       0 for Gaussian auto t,
                       1 for Gaussin fixed t */
    bool only = false;
    while ( (c = getopt_long(argc, argv,
//                     "vw:e:n:K:S:I:z:t:a:s:x:o",
                    "vw:e:n:K:S:I:z:t:o",
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
            case 'K':
                K = string_as_T<int>(string(optarg));
                break;
            case 'S':
                rho = string_as_T<float>(string(optarg));
                break;
            case 'I':
                Iter = string_as_T<int>(string(optarg));
                break;
            case 'z':
                spa = string_as_T<int>(string(optarg));
                break;
            case 't':
                fixed_t = string_as_T<float>(string(optarg));
                break;
//             case 'a':
//                 p_ann = string(optarg);
//                 break;
//             case 's':
//                 size = string_as_T<int>(string(optarg));
//                 break;
//             case 'x':
//                 exp = string_as_T<int>(string(optarg));
//                 break;
            case 'o':
                only = true;
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
    if (ped == string(""))
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
    if (K < 1)
    {
        KPROP_ERROR("invalid K for NN-Descent/NNF-Descent.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (rho > 1.0F || rho <= 0.0F)
    {
        KPROP_ERROR("invalid sampling rate for NN-Descent/NNF-Descent.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (Iter < 0)
    {
        KPROP_ERROR("invalid max number of iterations for NNF-Descent.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (spa < 0)
    {
        KPROP_ERROR("invalid max number of dimensions to sparsify each time.\n");
        printUsage();
        exit(EXIT_FAILURE);
    }
    if (fixed_t < -TOLERANCE_VALUE)
    {
        simtype = -1;
    }
    else if (fixed_t > TOLERANCE_VALUE)
    {
        simtype = 1;
    }
    else
    {
        simtype = 0;
    }

    if (verbose)
    {
        printf("# SYSTEM PARAMETERS:\n");
        printf("#      Path of working directory: %s\n",
                                             pwd.c_str());
        printf("#   Path of experiment directory: %s\n",
                                             ped.c_str());
        printf("#                   Dataset name: %s\n",
                                             dataset.c_str());
        printf("# NN-DESCENT / NNF-DESCENT PARAMETERS:\n");
        printf("#                 K of NN-Descent/NNF-Descent: %d\n", K);
        printf("#     Sampling rate of NN-Descent/NNF-Descent: %f\n", rho);
        printf("#          Max # of iterations of NNF-Descent: %d\n", Iter);
        printf("#   Max # of dimensions to sparsify each time: %d\n", spa);
        printf("#                             Similarity type: ");
        if (simtype == -1)
        {
            printf("linear similarity\n");
        }
        else if (simtype == 0)
        {
            printf("Gaussian (auto t)\n");
        }
        else if (simtype == 1)
        {
            printf("Gaussian (fixed t at %f)\n", fixed_t);
        }
        else
        {
            assert(0);
        }
//         printf("# KNN classification parameters:\n");
//         printf("#                    Path of annotation files: %s\n",
//                 p_ann.c_str());
//         printf("#                             Annotation size: %d\n", size);
//         printf("#                           Experiment number: %d\n", exp);
//         printf("#\n");
//         printf("#            Only experiment estimated graphs: %s\n",
//                 (only?"Yes":"No"));
        printf("#\n");
        fflush(stdout);
    }

    /* part 1: loading descriptors in KProp dvf format */
    if (verbose)
    {
        printf("# Loading descriptors... Warning: descriptors should be "
               "standardized in advance.\n");
        fflush(stdout);
    }

    string p_data_dvf = pwd + dataset + string(".dvf");
    DenMatSin desc(p_data_dvf, true); /* feature values stored here will
                                         not change */
    int nr_data = desc.getM();
    int dim     = desc.getN();

    /* loading true labels */
    if (verbose)
    {
        printf("# Loading true labels...\n");
        fflush(stdout);
    }
    string p_data_lab = ped + dataset + string(".matlab.lab");
    FILE* fp = fopen(p_data_lab.c_str(), "rt");
    assert(fp != NULL);
    int* true_lab = new int[nr_data];
    int nr_labels = -1; /* labels are 1,2,3,4,...,nr_labels */
    for (int n = 0; n < nr_data; ++n)
    {
        int c = fscanf(fp, "%d\n", true_lab + n);
        assert(c == 1);
        if (true_lab[n] > nr_labels)
            nr_labels = true_lab[n];

    }
    fclose(fp);

    /* loading true KNN */
    if (verbose)
    {
        printf("# Loading true KNNs...\n");
        fflush(stdout);
    }
    string p_data_knn_1 = ped + dataset + string(".knn.1");
    //string p_data_knn_2 = ped + dataset + string(".knn.2");
    DenMatSin true_knns(p_data_knn_1, true);
    //DenMatSin true_knns_dist(p_data_knn_2, true);
    assert(true_knns.getM() == nr_data);
    assert(true_knns.getN() > K);
    //assert(true_knns_dist.getM() == nr_data);
    //assert(true_knns_dist.getN() == true_knns.getN());
    printf("%d",true_knns.getM());
    printf("%d",true_knns.getN());
    
    /* no need to validate this KNN, because .knn.1 file is good! */

    /* note, for evaluation, use small_k, default K */
    int small_k = K;

    /* a matrix saving number of correct neighbors of each data point */
    DenMatSin nrCorrNeighbors(Iter+1, nr_data);
    /* if only, the first row will be all zeros */




    if (!only)
    {
        if (verbose)
        {
            printf("# true-KNN  correctness(%%)  edge-precision(%%)\n");
        }

        float correctness = compCorrectness(true_knns, true_lab, nr_data,
                                            10, nrCorrNeighbors, 0);  // put 10 instead of small_k
        //float ep = compEdgePrecision(true_knns, true_lab, nr_data, small_k);   /* commented by Arwa Wali to avoid segmentation fault */

        float varCorrNeighbors = 0.0F;
        float meanCorrNeighbors = 0.0F;
        for (int n = 0; n < nr_data; ++n)
        {
            meanCorrNeighbors += nrCorrNeighbors(0, n);
        }
        meanCorrNeighbors /= (float) nr_data;
        for (int n = 0; n < nr_data; ++n)
        {
            float temp = nrCorrNeighbors(0, n) - meanCorrNeighbors;
            varCorrNeighbors += temp*temp;
        }
        varCorrNeighbors /= (float) nr_data;

        //printf("0\t%.2f%%\t%.2f%%\t%f\n",correctness*100.0F, ep*100.0F, varCorrNeighbors);   /* commented by Arwa Wali to avoid segmentation fault */
        printf("0\t%.2f%%\t%%\t%f\n",correctness*100.0F, varCorrNeighbors);
        //printf("%d\t%.2f%%\t%.2f%%\t%f\n",Iter, correctness*100.0F, ep*100.0F, varCorrNeighbors); /* commented by Arwa Wali to avoid segmentation fault */
        printf("%d\t%.2f%%\t%%\t%f\n",Iter, correctness*100.0F, varCorrNeighbors);
        printf("\n");
        fflush(stdout);
    }


    /* prepare for NN-Descent / NNF-Descent */
    if (verbose)
    {
        printf("# Preparing for NN-Descent / NNF-Descent...\n");
        fflush(stdout);
    }
    Dataset<float> desc_for_nndes; /* feature values stored here will change */
    desc_for_nndes.load(&desc);
    OracleL2<Dataset<float> > oracle_for_nndes(desc_for_nndes);
    assert(desc_for_nndes.size() == nr_data);
    assert(desc_for_nndes.getDim() == dim);
    
    
    /* Edited By Arwa Wali Jun. 7, 2015 **********************/
     
     // Add string bits to save the features that has been sparsified by nnf-descent
     //vector<bitset<1000000> > flags(nr_data);
     vector<boost::dynamic_bitset<> > dyn_flags( nr_data, boost::dynamic_bitset<>(dim)) ;
    // initlize the flags to 1s
    for(int i=0; i< nr_data; i++){
       dyn_flags[i].set();
       //for(int j=0; j< dim; j++){
       //    if ( (float)  desc_for_nndes[i][j]== 0.0F)
       //    {
        //       dyn_flags[i][j]=0;
        //   }
       //} 
        
    }
     /*********************************************************/
     
    /* part 2: initial NN-Descent */
    int pre = 100; /* should be enough for a convergence */
    if (verbose)
    {
        printf("# Starting initial NN-Descent for at most %d iterations...\n",
                pre);
        fflush(stdout);
    }
    ///
    clock_t begin= clock();
    ///

    NNDescent<OracleL2<Dataset<float> > > 
        nndes(nr_data, K, rho, oracle_for_nndes, GRAPH_BOTH);

    float total = float(nr_data) * (nr_data - 1) / 2; /* brutal force
                                                         cost */
    float Delta = 0.001;
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

        float rec = 0.0F; /* recall */
        const vector<KNN>& curr_knns = nndes.getNN();
        for (int n = 0; n < nr_data; ++n)
        {
            rec += recall( &(true_knns(n, 1)), curr_knns[n], K );
        }
        rec /= (float) nr_data;

        if (verbose)
        {
            printf("# NN-Descent %d, UpdateRate %.2f%%, CostRate %.2f%%, "
                   "Recall %.2f%%\n",
                    it+1, update_rate*100.0F, cost_rate*100.0F,
                    rec*100.0F);
        }
        if (update_rate < Delta) break;
    }

    clock_t end = clock();
    double secs = double(end-begin)/CLOCKS_PER_SEC;
    cout << secs << "s for original nn-descent" << endl;

    /* make sure that NN-Descent generates a valid KNN */
    assert(validateKnn(nndes.getNN()));

    // Edited by Arwa Wali to Print KNN for NN-Descent before sparsification) */ 15-Nov-2020
   
    
       const vector<KNN> &nn1 = nndes.getNN();
       string file_name_orginal=ped+ dataset+ "_initialNNFK_Z_";
        std::stringstream sstm_orginal;
        sstm_orginal << file_name_orginal <<K<<"_"<<spa;
        string result_orginal = sstm_orginal.str();
        
        ofstream os_orginal(result_orginal.c_str());
        int i = 0;
        BOOST_FOREACH(KNN const &knn, nn1) {
            os_orginal << i++;
            BOOST_FOREACH(KNN::Element const &e, knn) {
                os_orginal << ' ' << e.key;
            }
            os_orginal << endl;
        }
        os_orginal.close();
        fflush(stdout);


   

    /* now the NN-Descent algorithm pauses */
    /* pause NN-Descent */
    nndes.preNnf();

    float max_graph_dist = -FLT_MAX;
    float min_graph_dist = FLT_MAX;
    float auto_t = -1.0F;

    if (simtype == -1) /* linear */
    {
        if (verbose) printf("# linear similarity\n");
        compMaxMinEdgeDist(nndes.getNN(), &max_graph_dist, &min_graph_dist);
    }
    else if (simtype == 0) /* Gaussian auto t */
    {
        if (verbose) printf("# Gaussian auto t\n");
        auto_t = compMeanEdgeDist(nndes.getNN());
        auto_t = 2*auto_t*auto_t;
    }
    else if (simtype == 1) /* Gaussian fixed t */
    {
        if (verbose) printf("# Gaussian t fixed at %f\n", fixed_t);
    }
    else
    {
        assert(0);
    }
    fflush(stdout);

    /* this is just a rough compared flag */
    
    /* This part commneted by Arwa Wali (No need) and to avoid Segmentation fault
    DenMatSin compared(nr_data, nr_data);

    printf("%d",compared.getM());
    printf("%d",compared.getN());

    

    const vector<KNN>& curr_knns = nndes.getNN();
    for (int n = 0; n < nr_data; ++n)
    {
        for (int k = 0; k < small_k; ++k)
        {   
            printf("%d\n", n);
 	        printf("%d\n", curr_knns[n][k].key);
            //compared(n, curr_knns[n][k].key)=1.0F;
            //compared(curr_knns[n][k].key, n)=1.0F;
            compared.set(n, curr_knns[n][k].key);
            compared.set(curr_knns[n][k].key, n); /* this might not be true, if pre = 0 */
     //   }
    //}

    /* part 3: NNF-Descent */
    if (verbose)
    {
        printf("# Starting NNF-Descent for at most %d iterations,\n"
               "# %d features sparsified in each iteration...\n", Iter, spa);
        printf("# Iter correctness(%%)  edge-precision(%%)\n");
    }

    srand( unsigned(time(0)) );
    vector<int> item_indices(nr_data);
    /* no need to initialize everytime in the each iteration */
    for (int n = 0; n < nr_data; ++n)
    {
        item_indices[n] = n;
    }




    /// new ///
    float avg_corr = 0.0F;
    float best_corr = -1.0F;
    int   best_corr_iter = -1;

    float avg_ep = 0.0F;
    float best_ep = -1.0F;
    int   best_ep_iter = -1;
    
    //int kmax=100;
    
    // Calculate r for each object for each feature, we can take sample of 100 objects or whole objects. Here I take 100 points, becuse the huge dataset
    //  /****   Compute the r=avg(w), which is k-NN distance for each object for each feature.
    // This is fixed a head
    
    // Edited By Arwa Wali
    
    vector<float> rs_avg_w(dim);
    int sample_size=100;
    const vector<KNN>& curr_knns = nndes.getNN();
    

    /* Generate a random order list for all data items */
    random_shuffle(item_indices.begin(), item_indices.end(), myrandom);
    #pragma omp parallel for
    for (int f=0; f<dim; f++){
        float r_j=0.0F; // sum the r for each object per feature
        //for (int n = 0; n < nr_data; ++n) /* for each data item in the random
          //                                 list */
        
        //for (int n = 0; n < 100; ++n) /* for each data item in the random list */

        //{
         //   int p = item_indices[n];
            // desc_for_nndes[p] and all other elements for feature f
        //    vector<float> de(nr_data);
         //   #pragma omp parallel for
       //     for (int k=0; k<nr_data; k++) {
       //         de[k]=abs(desc_for_nndes[p][f]-desc_for_nndes[k][f]);
       //     }
       //     // sort distances up to the first nonzero + kmax  // change to K (Arwa Wali)
       //     int nz = 0; // number of zeros
       //     for (int k=0; k<nr_data; k++) {if (de[k]==0) nz++;}
            //partial_sort(&de[0], &de[min(nz+kmax,nr_data)], &de[nr_data]);
       //     partial_sort(&de[0], &de[min(nz+K,nr_data)], &de[nr_data]);
            // evaluate ID(obs[i])
            //localID= evalID(&(de[nz]), min(kmax,nr_data-nz));
        //    r_j+= w_KNN_distance(&(de[nz]), min(K,nr_data-nz));
            
       // }
        //rs_avg_w[f]=(float) r_j/(float) nr_data;
        //rs_avg_w[f]=(float) r_j/(float) 100;
        for (int n = 0; n < sample_size; ++n)   /* for each data item in the random list */
        
        {
            int p = item_indices[n];
            // get the distance between each object and current K neighbor for feature f
            r_j+=abs(desc_for_nndes[p][f]-desc_for_nndes[curr_knns[p][K-1].key][f]);
            
            
        }
        //rs_avg_w[f]=(float) r_j/(float) nr_data;
        rs_avg_w[f]=(float) r_j/(float) sample_size;
    }
    float const_rs_avg_w=0.0F;
    for(int f=0; f<dim; f++){
        const_rs_avg_w+=rs_avg_w[f];
    }
    const_rs_avg_w=const_rs_avg_w/(float) dim;
    
    // print rs_avg_w for test
    
    /*for(int f=0; f<dim; f++){
     cout << rs_avg_w[f] << "  ";
     }
     cout << endl; */
    
    /*  prepare a nwe KNN graph as orginal to use for functional evaluation   */
    // make a copy of the current K-NN graph to use it for functional quality evaluation
    const vector<KNN> curr_knns_functional =nndes.getNN();
    cout << curr_knns_functional.size() << endl;

    
    // ********************************************************************


    for (int it = 0; it < Iter; ++it) /* each iteration is an iteration
                                         of NNF-Descent
                                         each iteration goes through all
                                         data items */

    {
        /* Generate a random order list for all data items */
        random_shuffle(item_indices.begin(), item_indices.end(), myrandom);

       
        clock_t total0 = 0;
        clock_t total = 0;
        clock_t total3 = 0;
        clock_t total4 = 0;
        for (int n = 0; n < nr_data; ++n) /* for each data item in the random
                                             list */
        {
            clock_t b1 = clock();
            /* current estimated KNN lists */
            const vector<KNN>& curr_knns = nndes.getNN();
            int p = item_indices[n]; /*  p is the index of the pivot point */

            /* s1. feature sparsification */
            /* rank all p's features in descending order of their local
             * Laplcain scores computed from its current K-NN (and R-NN?
             * Mike suggests using K-NN only now, otherwise, will need to
             * integrate the two sets of LLS?) */
            ifpair* sort_dim = new ifpair[dim];
            int nr_cand_dims = 0; /* check only for non-removed/non-sparsified
                                     dims */
            clock_t b2 = clock();
            /*
             Edited By Arwa Wali 
             We use ID per feature (Oussama'code) to find the rank of each feature per object
             */
            //vector<int> sf(dim);  // store the features' order
            //#pragma omp parallel for
            //for (int i=0; i<dim; i++)  sf[i]=i;
            
            /**********************
             * Select features using the whole dataset
             ************************/
            
            // Compute ID values
            //score.resize(dim);
            for(int f=0; f<dim; f++){
                if ( (float) fabs(desc_for_nndes[p][f]) > TOLERANCE_VALUE){
                    double localID;
                    // We compute distances between
                    // desc_for_nndes[p] and all other elements for feature f
                    vector<float> de(nr_data);
                    #pragma omp parallel for
                    for (int k=0; k<nr_data; k++) {
                        de[k]=abs(desc_for_nndes[p][f]-desc_for_nndes[k][f]);
                    }
                    // sort distances up to the first nonzero + kmax  // change to K (Arwa Wali)
                    int nz = 0; // number of zeros
                    for (int k=0; k<nr_data; k++) {if (de[k]==0) nz++;}
                    //partial_sort(&de[0], &de[min(nz+kmax,nr_data)], &de[nr_data]);
                    partial_sort(&de[0], &de[min(nz+K,nr_data)], &de[nr_data]);
                    // evaluate ID(obs[i])
                    //localID= evalID(&(de[nz]), min(kmax,nr_data-nz));
                    float w_f=w_KNN_distance(&(de[nz]), min(K,nr_data-nz));
                    localID= evalID(&(de[nz]), min(K,nr_data-nz));
                    
                    
                    
                    //sf[nr_cand_dims]=f;
                    //score[nr_cand_dims++] =localID; instead of sf and score
                    
                    /* Edite by Arwa Wali */
                    sort_dim[nr_cand_dims].index  = f;
                    sort_dim[nr_cand_dims++].weight =localID * pow ((const_rs_avg_w/w_f), localID) ;
                    
                }
                else{
                    dyn_flags[p][f]=0;
                }
                
            }
            // Features are sorted by increasing Scores sf[0] is the best, sf[dim] is the worst
            //sort(&sf[0],&sf[nr_cand_dims], compareScore);
            
            // Edited by arwa Wali to use sort_dim of Jishao to save Object'feature ID insead of LLS
            
            /* the larger value of the ID,  the worse of the feature */
            qsort(sort_dim, nr_cand_dims, sizeof(ifpair), cmp_ifpairs_d);
            
            /* Change the values of a few bad features to zero */
            //for (int d = KPROP_MIN(spa, nr_cand_dims)-1; d >=0; --d)
            //{
                /* do changing values for item p */
                //int dd = sf[d];
                //desc_for_nndes[p][dd] = 0.0F;
                
                
                /* Edited By Arwa Wali Jun. 7, 2015 */
                /* Also set the bit flag for each sparsifed dimention to 1 */
                //flags[p][dd]=1;
                //dyn_flags[p][dd]=1;
            //    /***********************************************************/
            //}
            
            for (int d = 0; d < KPROP_MIN(spa, nr_cand_dims); ++d)
            {
                /* do changing values for item p */
                int dd = sort_dim[d].index;
                desc_for_nndes[p][dd] = 0.0F;
                
                
                /* Edited By Arwa Wali Jun. 7, 2015 */
                /* Also set the bit flag for each sparsifed dimention to 1 */
                //flags[p][dd]=1;
                dyn_flags[p][dd]=0;
                /***********************************************************/
            }
            delete[] sort_dim;
            //sf.clear();
            /*
            for (int d = 0; d < dim; ++d)
            {
                if ( (float) fabs(desc_for_nndes[p][d]) > TOLERANCE_VALUE)
                                                    // non-zero dim only
                {
                    sort_dim[nr_cand_dims].index  = d;
                    if (simtype == -1)
                    {
                        sort_dim[nr_cand_dims++].weight =
                            llsLite(p, d, curr_knns[p], desc_for_nndes,
                                    max_graph_dist, min_graph_dist);
                    }
                    else if (simtype == 0)
                    {
                        sort_dim[nr_cand_dims++].weight =
                            llsLite(p, d, curr_knns[p], desc_for_nndes,
                                    auto_t);
                    }
                    else if (simtype == 1)
                    {
                        sort_dim[nr_cand_dims++].weight =
                            llsLite(p, d, curr_knns[p], desc_for_nndes,
                                    fixed_t);
                    }
                    else
                    {
                        assert(0);
                    }
                }
            }
            /* the larger value of the LLS, the worse of the feature */
            //qsort(sort_dim, nr_cand_dims, sizeof(ifpair), cmp_ifpairs_d);
            /* Change the values of a few bad features to zero */
            //for (int d = 0; d < KPROP_MIN(spa, nr_cand_dims); ++d)
            //{
                /* do changing values for item p */
                //int dd = sort_dim[d].index;
               // desc_for_nndes[p][dd] = 0.0F;
                
                
                /* Edited By Arwa Wali Jun. 7, 2015 */
                /* Also set the bit flag for each sparsifed dimention to 1 */
                //flags[p][dd]=1;
                //dyn_flags[p][dd]=1;
                /***********************************************************/
            //}
            //delete[] sort_dim;
            clock_t e2 = clock();
            total += e2-b2;

            /* s2. make current status of K-NN consistent */
            /* recompute the distances from p to its K-NN and R-NN
             * and re-order p's KNN list and p's RNN's KNN lists using the
             * new distance values */
            clock_t b3 = clock();
            
            // Edited by Arwa Wali by deleting compared, because it is not uded in nnfAdjust
            //nndes.nnfAdjust(p, compared);
            nndes.nnfAdjust(p);
            clock_t e3 = clock();
            total3 += e3-b3;
            

            clock_t b4 = clock();
            /* s3. An NN-Descent-like update for the pivot */
            // nndes.nnf(p, compared);
            nndes.nnf(p);
            clock_t e4 = clock();
            total4 += e4-b4;

            clock_t e1 = clock();
            total0 += e1 - b1;

            if (simtype == -1)
            {
                compMaxMinEdgeDist(nndes.getNN(), &max_graph_dist,
                        &min_graph_dist);
            }
            else if (simtype == 0)
            {
                auto_t = compMeanEdgeDist(nndes.getNN());
                auto_t = 2*auto_t*auto_t;
            }
            else if (simtype == 1)
            {
                /* do nothing */
            }
            else
            {
                assert(0);
            }
            
        }
        double secs = double(total0)/CLOCKS_PER_SEC;
        cout << secs << "s for one nnf-des" << endl;

        secs = double(total)/CLOCKS_PER_SEC;
        cout << secs << "s for one selection of nnf-des" << endl;

        secs = double(total3)/CLOCKS_PER_SEC;
        cout << secs << "s for one adjust of nnf-des" << endl;

        secs = double(total4)/CLOCKS_PER_SEC;
        cout << secs << "s for one update of nnf-des" << endl;

        // assert(validateKnn(nndes.getNN()));

        float correctness = compCorrectness(nndes.getNN(), true_lab, nr_data,
                                            10, nrCorrNeighbors, it+1);   // instead of small_k use 10
        //float ep          = compEdgePrecision(nndes.getNN(), true_lab, nr_data, small_k); /* commented by Arwa Wali to avoid segmentation fualt */
        float varCorrNeighbors = 0.0F;
        float meanCorrNeighbors = 0.0F;
        for (int n = 0; n < nr_data; ++n)
        {
            meanCorrNeighbors += nrCorrNeighbors(it+1, n);
        }
        meanCorrNeighbors /= (float) nr_data;
        for (int n = 0; n < nr_data; ++n)
        {
            float temp = nrCorrNeighbors(it+1, n) - meanCorrNeighbors;
            varCorrNeighbors += temp*temp;
        }
        varCorrNeighbors /= (float) nr_data;
        
        

        

        //printf("%d\t%.2f%%\t%.2f%%\t%f\n", it+1, correctness*100.0F, ep*100.0F, varCorrNeighbors);
        printf("%d\t%.2f%%\t%%\t%f\n", it+1, correctness*100.0F, varCorrNeighbors);  /* Edited by Arwa Wali to print only correctenss */
        fflush(stdout);
        // nrCorrNeighbors.print(10,0);


        /// new ///
        avg_corr += correctness;
        if (correctness > best_corr)
        {
            best_corr = correctness;
            best_corr_iter = it+1;
        }
        /* Commented By Arwa Wali to avoid segementation fault */
        /*
        avg_ep += ep;
        if (ep > best_ep)
        {
            best_ep = ep;
            best_ep_iter = it+1;
        } */

        // avg_knn += caccuracy;
        // if (caccuracy > best_knn)
        // {
        //     best_knn = caccuracy;
        //     best_knn_iter = it+1;
        // }
        // avg_1nn += caccuracy2;
        // if (caccuracy2 > best_1nn)
        // {
        //     best_1nn = caccuracy2;
        //     best_1nn_iter = it+1;
        // }
        ///
        /* Functional evaluation begian using */
        //const vector<KNN>& curr_knns = nndes.getNN();
        
        //float functional_performance=0.0F;
        /*int count_correct_neighbors=0;
        //#pragma omp parallel for
        for(int i=0; i<nr_data; i++){
            
            for(int k=0; k<K; k++){
                for(int j=0; j<K; j++){
                    if(curr_knns[i][k].key==curr_knns_functional[i][j].key){
                        count_correct_neighbors++;
                        break;
                        
                    }
                    
                }
            }
            
        }
        //cout << count_correct_neighbors << endl;
        //functional_performance+=(float) count_correct_neighbors;///(float) (K);
        functional_performance=(float) count_correct_neighbors/(float) (K*nr_data);
        printf("Functional Evaluation (|A ∩ B|/ (#data*K): \n");
        printf("%d\t%.2f\n", it+1, functional_performance*100.0F);
        fflush(stdout); */

           /* Edited By Arwa Wali
         Calculate the binary distances between each object its 100 NN
         
         */
        // Define the kNN indeces for each index using its binary distances
        int KNN_binary[nr_data][true_knns.getN()];
        //#pragma omp parallel for
        for(int iter2=0; iter2 < nr_data ; iter2++){
            boost::dynamic_bitset<> object_flag=dyn_flags[iter2];
            //cout<< "object 1" << endl;
            //cout<< object_flag << endl;
            ifpair* sort_index_bianry_distances = new ifpair[nr_data];
            for(int iter3=0; iter3< nr_data; iter3++){
                boost::dynamic_bitset<> object2_flag=dyn_flags[iter3];
                //cout<< "object 2" << endl;
                //cout<< object2_flag << endl;
                sort_index_bianry_distances[iter3].index=iter3;
                // calculate the hamming distance between two binary vectors using the respective features of object1
                float dis=0.0F;
                
                for(int d=0; d<dim; ++d){
                    //cout<<object_flag[d] <<endl;
                    //cout<<object2_flag[d]<<endl;
                    //cout<<" *****" <<object_flag[d]-object2_flag[d] <<endl;
                    if(object_flag[d]==1){
                        dis+=(float) pow(object_flag[d]-object2_flag[d],2);
                    
                    }
                    //if(object_flag[d]!=object2_flag[d]){
                      //  dis=dis+1.0F;
                       //cout<< dis << endl;
                    //}
                    
                }
                //dis=(float) sqrt(dis);
                //cout<<dis << endl;
                //boost::dynamic_bitset<> temp_flag=object_flag & object2_flag;
                
                //cout<< "object result" << endl;
                //cout<< temp_flag << endl;
                //sort_index_bianry_distances[iter3].weight=(float) temp_flag.count()/(float) object_flag.count();
                sort_index_bianry_distances[iter3].weight=(float) dis;
                //cout<< sort_index_bianry_distances[iter3].weight <<endl;
                //cout<< sort_index_bianry_distances[iter3].weight << endl;
            }
            // sort sort_index_bianry_distances according to th weight which contain the binary indexces and distances
            // the larger distances, the more close point to actual point
            qsort(sort_index_bianry_distances, nr_data, sizeof(ifpair), cmp_ifpairs_d);
            // copy the last indeces of sort_index_bianry_distances to KNN_binary
            int count=0;
            for( int c=nr_data-1; c>=max(nr_data-true_knns.getN(),0) ; c-- ){
                //cout<< sort_index_bianry_distances[c].index <<endl;
                KNN_binary[iter2][count]=sort_index_bianry_distances[c].index;
                count++;
            }
            delete[] sort_index_bianry_distances;
            
            
        }
        // Evaluate the KNN_bianary matrix
        
        // from k=small_k to k=true_knns.getN()
        cout << "Iteration "<< it+1 << endl;
        //#pragma omp parallel for
        int lis[6]={1, 5, 10, 15, 50, 100};
        for(int k=0; k<6; k++ ){
            int number_of_correct_class=0;
            //#pragma omp parallel for
           for(int b=0; b< nr_data; b++){
              for(int h=1; h<=lis[k]; h++){
                  //cout << true_lab[b] << " " << true_lab[KNN_binary[b][h]] << endl;
                  if(true_lab[b]==true_lab[KNN_binary[b][h]]){
                      number_of_correct_class=number_of_correct_class+1;
                  }
              }
            
           }
            //
           float accuracy=(float) number_of_correct_class/(float) (nr_data* lis[k]);
           cout << "Graph Accuracy for k= "<< lis[k] << "=" << accuracy <<endl ;
        }
         // We evaluating the precision int term of how many points from the actual distance
        // appear in the binary knn graph
        /*const vector<KNN>& curr_knns = nndes.getNN();
        for(int k=10; k<=100; k=k+10){
            int number_of_match=0;
            for(int b=0; b<nr_data; b++){
                for(int h=1; h<=10; h++){
                    for(int g=1; g<=k; g++){
                        if(curr_knns[b][h].key==KNN_binary[b][g]){
                            number_of_match=number_of_match+1;
                        }
                    }
                }
            }
            float precision=(float) number_of_match/(float) (nr_data* 10);
            cout << "Graph Precision for k= "<< k << "=" << precision <<endl ;
            
        }*/
        /*
        for(int k=small_k; k<=true_knns.getN(); k=k+k ){
            int number_of_correct_class=0;
            //#pragma omp parallel for
           for(int b=0; b< nr_data; b++){
              for(int h=1; h<=k; h++){
                  //cout << true_lab[b] << " " << true_lab[KNN_binary[b][h]] << endl;
                  if(true_lab[b]==true_lab[KNN_binary[b][h]]){
                      number_of_correct_class=number_of_correct_class+1;
                  }
              }
            
           }
            //
           float accuracy=(float) number_of_correct_class/(float) (nr_data* k);
           cout << "Graph Accuracy for k= "<< k << "=" << accuracy <<endl ;
        } */
    }
    /// new ///
     //Edited by Arwa to print thr NN-Desent after sparsification  15-Nov-2020
     
     const vector<KNN> &nn_sparse = nndes.getNN();
     string file_name_sparse=ped+ dataset+ "_NNFK_Z_";
     std::stringstream sstm_sparse;
     sstm_sparse << file_name_sparse <<K<<"_"<<spa;
     string result_sparse = sstm_sparse.str();
      
     
     ofstream os_sparse(result_sparse.c_str());
      i = 0;
     BOOST_FOREACH(KNN const &knn, nn_sparse) {
         os_sparse << i++;
         //printf("%d", i);
         BOOST_FOREACH(KNN::Element const &e, knn) {
             os_sparse << ' ' << e.key;
             //printf("%d%t", e.key);
         }
         os_sparse << endl;
         //printf("\n");
     }
     os_sparse.close();
     fflush(stdout);
    
    // print the KNN_binary after sparsiifcation 15-Nov-2020
    
    string file_name_binary=ped+ dataset+ "_Bianry_NNFK_Z_";
    std::stringstream sstm_bianry;
    sstm_bianry << file_name_binary <<K<<"_"<<spa;
    string result_bianry = sstm_bianry.str();
     
    
    ofstream os_bainary(result_bianry.c_str());
    int KNN_binary[nr_data][true_knns.getN()];
    for( int i=0; i<nr_data; i++){
        os_bainary << i++;
        for( int j=1; j<=true_knns.getN();j++ ){
            os_bainary  << ' ' << KNN_binary[i][j];
        }
        os_bainary << endl;
    }
    os_bainary.close();
    fflush(stdout);
   
    
    
    // Print the data after specification Arwa Wali
    // save spresified data to
    
    string file_name=ped+ dataset+ "_K_z2_";
    std::stringstream sstm;
    sstm << file_name << K<<"_"<<spa;
    string result = sstm.str();
    
    ofstream fs;
    fs.open(result.c_str(), ios::out);
    //myfile.open (file_name, ios::out);
    
    for( int i=0; i<nr_data; i++){
        for( int j=0; j<dim; j++){
            //cout<<desc_for_nndes[i][j]<<" ";
            fs <<desc_for_nndes[i][j]<<" ";
        }
        //cout<<"\n";
        fs <<"\n";
    }
    fs.close();
    fflush(stdout);
    
    /*  Edit by Arwa Wali to save all bits flag to file */
    
    // Print the data after specification Arwa Wali
    // save spresified data to
    
    file_name=ped+ dataset+ "flags_K_z2_";
    sstm.str(std::string());
    sstm << file_name << K<<"_"<<spa;
    result = sstm.str();
    
    //ofstream fs;
    fs.open(result.c_str(), ios::out);
    //myfile.open (file_name, ios::out);
    
    for( int i=0; i<nr_data; i++){
        for( int j=0; j<dim; j++){
            //cout<<desc_for_nndes[i][j]<<" ";
            //fs <<flags[i][j]<<" ";
            fs <<dyn_flags[i][j]<<" ";
        }
        //cout<<"\n";
        fs <<"\n";
    }
    fs.close();
    fflush(stdout);
    /*************************************************/



    avg_corr /= (float) Iter;
    //avg_ep   /= (float) Iter;   /* Commented By Arwa Wali to avoid segmentation fault */
    // avg_knn /= (float) Iter;
    // avg_1nn /= (float) Iter;
//     printf("# AVGTP: %.2f%%, BESTTP: %.2f%%(%d), AVGKNN: %.2f%%, BESTKNN: %.2f%%(%d), AVG1NN: %.2f%%, BEST1NN: %.2f%%(%d)\n",
//             avg_corr*100.0F, best_corr*100.0F, best_corr_iter,
//             avg_knn*100.0F, best_knn*100.0F, best_knn_iter,
//             avg_1nn*100.0F, best_1nn*100.0F, best_1nn_iter);
    //printf("# AVGTP: %.2f%%, BESTTP: %.2f%%(%d), AVGEP: %.2f%%, BESTEP: %.2f%%(%d)\n",
      //      avg_corr*100.0F, best_corr*100.0F, best_corr_iter,
        //    avg_ep*100.0F,   best_ep*100.0F,   best_ep_iter);
    /* re written by Arwa Wali and deltete ep value to avoid segementation fault */
    printf("# AVGTP: %.2f%%, BESTTP: %.2f%%(%d)\n",
           avg_corr*100.0F, best_corr*100.0F, best_corr_iter);
    ///

    delete[] true_lab;

//     /**********************/
//     for (int it = 0; it <= Iter; ++it)
//     {
//         float* ptr = &(nrCorrNeighbors(it, 0));
//         qsort(ptr, nr_data, sizeof(float), cmp_fl_i);
//     }
// 
//     // nrCorrNeighbors.print(10,0);
// 
//     DenMatSin nrCorrNeighborsT(nr_data, Iter+1);
//     nrCorrNeighborsT.transpose(nrCorrNeighbors);
//     // nrCorrNeighborsT.print(10,0);
//     nrCorrNeighborsT.save(string("./num_corr_nn.dvf"), true, 0);
//     /**********************/




    if (verbose)
    {
        printf("# Done.\n\n");
    }
    return 0;
}
