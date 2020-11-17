/* KProp.cc */

/* Implements KProp.h.
 * Jichao Sun (js87@njit.edu)
 *
 * June 12, 2013
 *   variant KProp --> OriKProp to avoid conflict
 *   added iterative() to compute X=P3*X+B
 *
 * June 7, 2013
 *   read descriptor files back only when using SW-KProp+
 *   use sim() for the similarity function of SW-KProp and SW-KProp+
 *
 * June 6, 2013 Parameter names changed to be consistent with the paper
 *     df --> beta
 *     af --> alpha
 *   Bestmatch method(s) removed 
 *
 * June 5, 2013 Initialized based on KProp.cc (last modified on Mar 14,
 * 2013) from kprop.6
 *
 */

#include "KProp.h"
#include <cstdio>
#include <cfloat>   /* for FLT_MAX */
#include <ctime>    /* for random number */
#include <cstdlib>  /* for qsort() and exit() */
#include <cmath>    /* for some math function */
#include <cassert>
#include <sys/stat.h> /* to check whether a file exists */
#include <string>
using std::string;
#include <fstream>
using std::ifstream;
#include <vector>
using std::vector;
#include <stack>
using std::stack;
#include <sstream>
using std::istringstream;
#include "global.h"
#include "DenMatSin.h"
#include "nist_spblas/nist_spblas.h"
using namespace NIST_SPBLAS;

namespace kprop
{
    const char* KPROP_VAR[] = {"KProp",      /* KProp */
                               "SW-KProp",   /* SWKProp */
                               "SW-KProp+"}; /* SWKPropPlus */
}

kprop::KProp::KProp() { init(); }
kprop::KProp::KProp(const KPropConf& cfg)
{
    init();
    /* We defer the validation of configuration */
    config_ = cfg;
}
kprop::KProp::~KProp()
{
    if (config_.verbose)
    {
        printf("> Cleaning up... ");
        fflush(stdout);
    }
    delete knn_2_;       knn_2_       = NULL;
    delete knn_1_;       knn_1_       = NULL;
    delete descs_;       descs_       = NULL;
    delete label_index_; label_index_ = NULL;
    delete key_index_;   key_index_ = NULL;
    if (labels_ != NULL)
    {
        for (int i = 0; i < (int) labels_->size(); i++)
        {
            delete (*labels_)[i].data_indices;
            (*labels_)[i].data_indices = NULL;
        }
    }
    delete labels_;      labels_ = NULL;
    delete data_;        data_   = NULL;
    if (config_.verbose)
    {
        printf("done.\n");
        fflush(stdout);
    }
}
void kprop::KProp::start()
{
    if (config_.run)
        run();
    else
        annotate();
}
void kprop::KProp::init()
{
    config_.verbose   = false;
    config_.variant   = OriKProp;
    config_.pwd       = string("");
    config_.dataset   = string("");
    config_.ped       = string("");
    config_.p_ann     = string("");
    config_.run       = false;
    config_.ann_sizes = NULL;
    config_.nr_exp    = 0;
    config_.k         = 0;
    config_.beta      = 0.0F;
    config_.alpha     = 0.0F;
    config_.rd        = 0.0F;
    config_.tc        = 0.0F;
    config_.delta     = 0.0F;
    config_.max_iter  = 0;

    data_        = NULL;
    labels_      = NULL;
    key_index_   = NULL;
    label_index_ = NULL;
    descs_       = NULL;
    knn_1_       = NULL;
    knn_2_       = NULL;
    max_knn_2_   = 0.0F;
    min_knn_2_   = 0.0F;
}
/* initData() initializes data items (data_.key) and
 * computes the data item-->data index map (key_index_)
 *
 * afterward status (initialized (+), not initialized (-)):
 *   data_
 *     data_.key           (+)
 *     data_.label_index   (-)
 *     data_.annotated     (-)
 *     data_.assignment    (-)
 *   key_index_            (+)
 */
void kprop::KProp::initData()
{
    string p_data_key = config_.pwd + config_.dataset + string(".key");
    if (config_.verbose)
    {
        printf("> Initializing data items...\n");
        fflush(stdout);
    }
    ifstream ifs_data_key(p_data_key.c_str());
    if (ifs_data_key == NULL)
    {
        KPROP_ERROR("could not open unique key file '%s' to read.\n",
                    p_data_key.c_str());
        exit(EXIT_FAILURE);
    }
    string line;
    if (!getline(ifs_data_key, line))
    {
        KPROP_ERROR("corrupted header of unique key file '%s'.\n",
                    p_data_key.c_str());
        exit(EXIT_FAILURE);
    }
    trim(line);
    int nr_data = string_as_T<int>(line);
    if (nr_data <= 0)
    {
        KPROP_ERROR("invalid number of data items defined "
                    "in unique key file '%s'.\n", p_data_key.c_str());
        exit(EXIT_FAILURE);
    }
    vector<Datum>* data      = new vector<Datum>(nr_data);
    simap*         key_index = new simap;
    for (int i = 0; i < nr_data; i++)
    {
        if (!getline(ifs_data_key, line))
        {
            KPROP_ERROR("corrupted unique key file '%s'.\n",
                        p_data_key.c_str());
            exit(EXIT_FAILURE);
        }
        trim(line);
        if (key_index->find(line) != key_index->end())
        {
            KPROP_ERROR("duplicated key '%s' found on line %d in "
                        "unique key file '%s'.\n",
                        line.c_str(), i+1, p_data_key.c_str());
            exit(EXIT_FAILURE);
        }
        (*key_index)[line]     = i;
        (*data)[i].key         = line;
        (*data)[i].label_index = -1;    /* not initialized */
        (*data)[i].annotated   = false; /* not initialized */
        (*data)[i].assignment  = -1;    /* not initialized */
    }
    ifs_data_key.close();

    if (data_) delete data_;
    data_  = data;
    if (key_index_) delete key_index_;
    key_index_ = key_index;
    if (config_.verbose)
    {
        printf("  #(data items initialized): %d\n", nr_data);
        fflush(stdout);
    }
}
/* initLabels() initializes data items (data_.label_index),
 * labels (labels_.text, labels_.data_indices)
 * and computes label-->label index map (label_index_)
 *
 * afterward status (initialized (+), not initialized (-)):
 *   data_
 *     data_.key           (+)
 *     data_.label_index   (+)
 *     data_.annotated     (-)
 *     data_.assignment    (-)
 *   labels_
 *     labels_.text                    (+)
 *     labels_.data_indices            (+)
 *     labels_.nr_annotated            (-)
 *     labels_.nr_correct_assignments  (-)
 *   key_index_      (+)
 *   label_index_    (+)
 */
void kprop::KProp::initLabels()
{
    string p_data_lab = config_.pwd + config_.dataset + string(".lab");
    if (config_.verbose)
    {
        printf("> Initializing ground truth labels...\n");
        fflush(stdout);
    }
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
    int nr_data = string_as_T<int>(line);
    if (nr_data != (int)data_->size())
    {
        KPROP_ERROR("inconsistent number of data items defined "
                    "in ground truth label file '%s'.\n",
                    p_data_lab.c_str());
        exit(EXIT_FAILURE);
    }
    vector<Label>* labels      = new vector<Label>;
    simap*         label_index = new simap;
    /* we defer put data_.label_index */
    int* lis = new int [nr_data];
    for (int i = 0; i < nr_data; i++)
    {
        if (!getline(ifs_data_lab, line))
        {
            KPROP_ERROR("corrupted ground truth label file '%s'.\n",
                        p_data_lab.c_str());
            exit(EXIT_FAILURE);
        }
        trim(line);
        simap_cit cit = label_index->find(line);
        int li; /* label index */
        if (cit == label_index->end()) /* a new label */
        {
            Label tmp_label = {line, (new vector<int>), 0, 0};
            tmp_label.data_indices->push_back(i);
            labels->push_back(tmp_label);
            li = labels->size() - 1;
            (*label_index)[line] = li;
        }
        else
        {
            li = cit->second;
            (*labels)[li].data_indices->push_back(i);
        }
        lis[i] = li;
    }
    ifs_data_lab.close();
    for (int i = 0; i < nr_data; ++i)
    {
        (*data_)[i].label_index = lis[i];
    }
    delete[] lis;

    if (label_index_) delete label_index_;
    label_index_ = label_index;
    if (labels_) delete labels_;
    labels_ = labels;

    if (config_.verbose)
    {
        printf("  #(unique ground truth labels): %d\n",
               (int) labels_->size());
        fflush(stdout);
    }
}
/* annotate()
 * afterward status (initialized (+), not initialized (-)):
 *   data_
 *     data_.key           (+)
 *     data_.label_index   (+)
 *     data_.annotated     (+)
 *     data_.assignment    (-)
 *   labels_
 *     labels_.text                    (+)
 *     labels_.data_indices            (+)
 *     labels_.nr_annotated            (+)
 *     labels_.nr_correct_assignments  (-)
 *   key_index_      (+)
 *   label_index_    (+)
 */
void kprop::KProp::annotate()
{
    /* level 1 */
    initData();
    initLabels();
    /* level 2 */
    /* 2-0 produces ground truth label file for matlab code */
    string p_data_matlab_lab = config_.ped + config_.dataset + ".matlab.lab";
    struct stat buffer;
    if (stat(p_data_matlab_lab.c_str(), &buffer) == -1) /* not existing */
    {
        FILE* fp = fopen(p_data_matlab_lab.c_str(), "wt");
        if (fp == NULL)
        {
            KPROP_ERROR("could not open ground truth label file for MATLAB code '%s' "
                        "to write.\n", p_data_matlab_lab.c_str());
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < (int)data_->size(); i++)
        {
            /* matlab style: starting from 1 */
            fprintf(fp, "%d\n", (*data_)[i].label_index + 1);
        }
        fclose(fp);
    }

    /* 2-1 pre-annotating */
    int nr_data   = (int) data_->size();

    int nr_ann = (int) config_.ann_sizes->size();
    int nr_exp    = config_.nr_exp;
    /* check configurations */
    if ((nr_ann<=0)
            ||
        (nr_exp<=0))
    {
        KPROP_ERROR("invalid annotation size(s) or number of experiments "
                    "per annotation size.\n");
        exit(EXIT_FAILURE);
    }
    if (config_.verbose)
    {
        printf("> Pre-annotating data items using the following %d annotation "
                "sizes: ", nr_ann);
        for (int i = 0; i < (int) config_.ann_sizes->size(); i++)
        {
            printf("%d ", (*(config_.ann_sizes))[i]);
        }
        printf("\n");
        printf("  #(experiments per annotation size): %d\n", nr_exp);
        fflush(stdout);
    }
    for (int i = 0; i < (int) config_.ann_sizes->size(); i++)
    {
        if ( (*(config_.ann_sizes))[i] <= 0 )
        {
            KPROP_ERROR("invalid annotation size(s).\n");
            exit(EXIT_FAILURE);
        }
    }

    srand(time(NULL));
    int ann_count = 0;
    for (int ii = 0; ii < (int) config_.ann_sizes->size(); ii++)
    {
        int i = (*(config_.ann_sizes))[ii]; /* i is the ann size */
        for (int j = 0; j < nr_exp; ++j) /* j+1 is the experiment index */
        {
            /* pre-annotate data set */
            if (config_.verbose)
            {
                printf("  > Annotation %d/%d, experiment %d/%d, "
                       "ann size %d\n", (ann_count+1), nr_ann,
                       (j+1), nr_exp, i);
                fflush(stdout);
            }
            annotate(i);
            string p_data_ann = config_.ped + config_.p_ann
                                + config_.dataset + "."
                                + T_as_string(i) + "_"
                                + T_as_string(j+1) + ".ann";
            string p_data_matlab_ann = config_.ped + config_.p_ann
                                    + config_.dataset + ".matlab."
                                    + T_as_string(i) + "_"
                                    + T_as_string(j+1) + ".ann";
            FILE* fp1 = fopen(p_data_ann.c_str(), "wt");
            FILE* fp2 = fopen(p_data_matlab_ann.c_str(), "wt");
            if (fp1 == NULL)
            {
                KPROP_ERROR("could not open annotation file '%s' "
                            "to write.\n", p_data_ann.c_str());
                exit(EXIT_FAILURE);
            }
            if (fp2 == NULL)
            {
                KPROP_ERROR("could not open annotation file '%s' for MATLAB code "
                            "to write.\n", p_data_matlab_ann.c_str());
                exit(EXIT_FAILURE);
            }
            for (int k = 0; k < nr_data; k++)
            {
                if ((*data_)[k].annotated)
                {
                    fprintf(fp1, "%s\n", ((*data_)[k].key).c_str());
                    fprintf(fp2, "%d\n", k+1); /* matlab style:
                                                 starting from 1 */
                }
            }
            fclose(fp2);
            fclose(fp1);
            rollback(); /* rollback for next run */
        }
        ann_count ++;
    }
}
int kprop::KProp::annotate(int ann_size)
{
    if (ann_size <= 0)
    {
        KPROP_ERROR("invalid annotation size.\n");
        exit(EXIT_FAILURE);
    }
    int nr_labels = (int) labels_->size();
    int count     = 0; 
    for (int i = 0; i < nr_labels; i++)
    {
        int nr_data_per_label = (int)((*labels_)[i].data_indices->size());
        if (ann_size >= nr_data_per_label)
        {
            /* pre-annotate all data items with this label */
            KPROP_WARN( "all data items with ground truth label '%s' "
                        "will be pre-annotated.\n",
                        ((*labels_)[i].text).c_str());
            for (int j = 0; j < nr_data_per_label; j++)
            {
                annotateOne( (*((*labels_)[i].data_indices))[j] );
            }
            count += nr_data_per_label;
            continue;
        }
        /* randomly select ann_size data items */
        i2pair* i2pairs = new i2pair[nr_data_per_label];
        for (int j = 0; j < nr_data_per_label; j++)
        {
            i2pairs[j].index = (*((*labels_)[i].data_indices))[j];
            i2pairs[j].weight = rand();
        }
        qsort(i2pairs, nr_data_per_label, sizeof(i2pair), cmp_i2pairs_i);
        for (int j = 0; j < ann_size; j++)
        {
            annotateOne( i2pairs[j].index );
        }
        count += ann_size;
        delete[] i2pairs;
    }
    return count;
}
int kprop::KProp::annotate(const string& ann_file)
{
    ifstream in(ann_file.c_str());
    if (in == NULL)
    {
        KPROP_ERROR("could not open annotation file '%s' to read.\n",
                    ann_file.c_str());
        exit(EXIT_FAILURE);
    }

    int count = 0;
    string line;
    while (getline(in, line))
    {
        trim(line);
        if (line == string("")) continue;
        simap_cit cit = key_index_->find(line);
        if (cit == key_index_->end())
        {
            KPROP_ERROR("data item '%s' not found.\n", line.c_str());
            exit(EXIT_FAILURE);
        }
        annotateOne(cit->second);
        count++;
    }
    in.close();
    return count;
}
void kprop::KProp::annotateOne(int data_index)
{
    if (data_index < 0 || data_index >= (int) data_->size())
    {
        KPROP_ERROR("data index out of bound.\n");
        exit(EXIT_FAILURE);
    }
    (*data_)[data_index].annotated = true;
    int label_index = (*data_)[data_index].label_index;
    if (label_index < 0 || label_index >= (int)labels_ -> size())
    {
        /* this should not happen */
        KPROP_ERROR("label index out of bound.\n");
        exit(EXIT_FAILURE);
    }
    (*labels_)[label_index].nr_annotated++;
}
/* run()
 * afterward status (initialized (+), not initialized (-)):
 *   data_
 *     data_.key           (+)
 *     data_.label_index   (+)
 *     data_.annotated     (+)
 *     data_.assignment    (+/-)
 *   labels_
 *     labels_.text                    (+)
 *     labels_.data_indices            (+)
 *     labels_.nr_annotated            (+)
 *     labels_.nr_correct_assignments  (+/-)
 *   key_index_      (+)
 *   label_index_    (+)
 */
void kprop::KProp::run()
{
    /* level 1 */
    initData();
    initLabels();
	
    /* level 2 */
    int nr_data   = (int) data_->size();
    int nr_labels = (int) labels_->size();
    int nr_ann = (int) config_.ann_sizes->size();
    int nr_exp    = config_.nr_exp;
    /* check configurations */
    if ((nr_ann<=0)
            ||
        (nr_exp<=0))
    {
        KPROP_ERROR("invalid annotation size(s) or number of experiments "
                    "per annotation size.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < (int) config_.ann_sizes->size(); i++)
    {
        if ( (*(config_.ann_sizes))[i] <= 0 )
        {
            KPROP_ERROR("invalid annotation size(s).\n");
            exit(EXIT_FAILURE);
        }
    }
    /* optionally read descriptors back */
    if (config_.variant == SWKPropPlus) /* SW-KProp+ needs descriptors */
    {
        string p_data_dvf = config_.pwd + config_.dataset + string(".dvf");
        descs_            = new DenMatSin(p_data_dvf, true);
        int dim           = descs_->getN();
        if (nr_data != descs_->getM())
        {
            KPROP_ERROR("inconsistent number of data items defined in "
                        "dense vector file '%s'.\n", p_data_dvf.c_str());
            exit(EXIT_FAILURE);
        }
        if (dim < 1)
        {
            KPROP_ERROR("invalid descriptor dimension.\n");
            exit(EXIT_FAILURE);
        }
    }

    /* create seperate recall/accuracy matrices for different methods (if any):
     * the last column of recall matrix is to store average recalls
     * the last two columns of accuracy matrix are to store average
     * accuracies, and standard error of mean (sem) 
     */
    /* for KProp method */
    DenMatSin *kprop_recalls    = new DenMatSin(nr_ann, nr_exp+1);
    DenMatSin *kprop_accuracies = new DenMatSin(nr_ann, nr_labels*nr_exp+2);

    if (config_.verbose)
    {
        printf("> Starting propagation using %d annotation files...\n",
               nr_ann);
        fflush(stdout);
    }
    string p_data_knn_1 = config_.ped + config_.dataset + string(".knn.1");
    string p_data_knn_2 = config_.ped + config_.dataset + string(".knn.2");
    knn_1_ = new DenMatSin(p_data_knn_1,  true); /*ascii*/
    knn_2_ = new DenMatSin(p_data_knn_2,  true); /*ascii*/
    if (knn_1_->getM() != (int) data_->size())
    {
        KPROP_ERROR("inconsistent number of data items defined in "
                    "kNN matrix file '%s'.\n", p_data_knn_1.c_str());
        exit(EXIT_FAILURE);
    }
    if (knn_2_->getM() != (int) data_->size())
    {
        KPROP_ERROR("inconsistent number of data items defined in "
                    "kNN matrix file '%s'.\n", p_data_knn_2.c_str());
        exit(EXIT_FAILURE);
    }
    if (knn_2_->getN() != knn_1_->getN())
    {
        KPROP_ERROR("inconsistent number of neighbors defined in "
                    "kNN matrix file '%s'.\n", p_data_knn_2.c_str());
        exit(EXIT_FAILURE);
    }
    if (config_.k < 1 || config_.k > knn_1_->getN() - 1)
    {
        KPROP_ERROR("invalid parameter k %d.\n", config_.k);
        exit(EXIT_FAILURE);
    }
    if (config_.variant == SWKProp || config_.variant == SWKPropPlus)
    {
        if (config_.alpha < 1.0F)
        {
            KPROP_ERROR("invalid amplifying factor %f.\n", config_.alpha);
            exit(EXIT_FAILURE);
        }
    }
    if (config_.variant == SWKPropPlus)
    {
        if (config_.rd <= 0.0F || config_.rd > 1.0F)
        {
            KPROP_ERROR("invalid refined dimension factor %f.\n", config_.rd);
            exit(EXIT_FAILURE);
        }
        if (config_.tc <= 0.0F || config_.tc > 1.0F)
        {
            KPROP_ERROR("invalid top-to-check factor %f.\n", config_.tc);
            exit(EXIT_FAILURE);
        }
    } 
    if (config_.beta <= 0.0F || config_.beta >= 1.0F)
    {
        KPROP_ERROR("invalid damping factor %f.\n", config_.beta);
        exit(EXIT_FAILURE);
    }
    if (config_.delta <= 0.0F)
    {
        KPROP_ERROR("invalid tolerance value %f.\n", config_.delta);
        exit(EXIT_FAILURE);
    }
    if (config_.max_iter < 1)
    {
        KPROP_ERROR("invalid max number of iterations %d.\n",
                    config_.max_iter);
        exit(EXIT_FAILURE);
    }

    /* compute max_knn_2 and min_knn_2 */
    /* they will be re-computed for SW-KProp+ */
    max_knn_2_ = 0.0F;
    min_knn_2_ = FLT_MAX;
    for (int i = 0; i < nr_data; i++)
    {
        for (int j = 1; j <= config_.k; j++)
        {
            float dist = (*knn_2_)(i,j);
            if (dist > max_knn_2_) max_knn_2_ = dist;
            if (dist < min_knn_2_) min_knn_2_ = dist;
        }
    }

    int ann_count = 0;
    for (int ii = 0; ii < (int) config_.ann_sizes->size(); ii++)
    {
        int i = (*(config_.ann_sizes))[ii];
        for (int j = 0; j < nr_exp; j++) /* j+1 is the experiment index */
        {
            string p_data_ann = config_.ped + config_.p_ann
                                + config_.dataset + "."
                                + T_as_string(i) + "_"
                                + T_as_string(j+1) + ".ann";
            /* pre-annotate data set */
            if (config_.verbose)
            {
                printf("  > Annotation %d/%d and Experiment %d/%d\n",
                       ann_count+1, nr_ann, (j+1), nr_exp);
                fflush(stdout);
            }
            annotate(p_data_ann);
            /* do kprop */
            kprop();
            compRecall  ((*kprop_recalls),    ann_count, j);
            compAccuracy((*kprop_accuracies), ann_count ,j);
            /* remove assignments, keep initial labeling
             * for next method (if any) */
            clearAssignments();
            /* end of kprop */

            /* rollback for next run, remove assignments and
             * initial labeling */
            rollback();
            /* flush output in each run */
            fflush(stdout);
        }

        /* code for result of each run can be written here */

        ann_count ++;
    }

    compAverage( *kprop_recalls, *kprop_accuracies );
    
    /* output results */
    if (config_.verbose)
    {
        printf("> Results:\n");
    }

    // printf("  # %s\n", KPROP_VAR[config_.variant]);
    // results(*kprop_recalls, *kprop_accuracies);
    resultsSim(*kprop_recalls, *kprop_accuracies);
    // printf("\n");
    delete kprop_recalls;
    delete kprop_accuracies;

}

void kprop::KProp::start(DenMatSin* knn_1, DenMatSin* knn_2)
{
    /* level 1 */
    initData();
    initLabels();
	
    /* level 2 */
    int nr_data   = (int) data_->size();
    int nr_labels = (int) labels_->size();
    int nr_ann = (int) config_.ann_sizes->size();
    int nr_exp    = config_.nr_exp;
    /* check configurations */
    if ((nr_ann<=0)
            ||
        (nr_exp<=0))
    {
        KPROP_ERROR("invalid annotation size(s) or number of experiments "
                    "per annotation size.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < (int) config_.ann_sizes->size(); i++)
    {
        if ( (*(config_.ann_sizes))[i] <= 0 )
        {
            KPROP_ERROR("invalid annotation size(s).\n");
            exit(EXIT_FAILURE);
        }
    }
    /* optionally read descriptors back */
    if (config_.variant == SWKPropPlus) /* SW-KProp+ needs descriptors */
    {
        string p_data_dvf = config_.pwd + config_.dataset + string(".dvf");
        descs_            = new DenMatSin(p_data_dvf, true);
        int dim           = descs_->getN();
        if (nr_data != descs_->getM())
        {
            KPROP_ERROR("inconsistent number of data items defined in "
                        "dense vector file '%s'.\n", p_data_dvf.c_str());
            exit(EXIT_FAILURE);
        }
        if (dim < 1)
        {
            KPROP_ERROR("invalid descriptor dimension.\n");
            exit(EXIT_FAILURE);
        }
    }

    /* create seperate recall/accuracy matrices for different methods (if any):
     * the last column of recall matrix is to store average recalls
     * the last two columns of accuracy matrix are to store average
     * accuracies, and standard error of mean (sem) 
     */
    /* for KProp method */
    DenMatSin *kprop_recalls    = new DenMatSin(nr_ann, nr_exp+1);
    DenMatSin *kprop_accuracies = new DenMatSin(nr_ann, nr_labels*nr_exp+2);

    if (config_.verbose)
    {
        printf("> Starting propagation using %d annotation files...\n",
               nr_ann);
        fflush(stdout);
    }
    // string p_data_knn_1 = config_.ped + config_.dataset + string(".knn.1");
    // string p_data_knn_2 = config_.ped + config_.dataset + string(".knn.2");
    // knn_1_ = new DenMatSin(p_data_knn_1,  true); /*ascii*/
    // knn_2_ = new DenMatSin(p_data_knn_2,  true); /*ascii*/
    knn_1_ = knn_1;
    knn_2_ = knn_2;
    if (knn_1_->getM() != (int) data_->size())
    {
        // KPROP_ERROR("inconsistent number of data items defined in "
        //             "kNN matrix file '%s'.\n", p_data_knn_1.c_str());
        exit(EXIT_FAILURE);
    }
    if (knn_2_->getM() != (int) data_->size())
    {
        // KPROP_ERROR("inconsistent number of data items defined in "
        //             "kNN matrix file '%s'.\n", p_data_knn_2.c_str());
        exit(EXIT_FAILURE);
    }
    if (knn_2_->getN() != knn_1_->getN())
    {
        // KPROP_ERROR("inconsistent number of neighbors defined in "
        //             "kNN matrix file '%s'.\n", p_data_knn_2.c_str());
        exit(EXIT_FAILURE);
    }
    if (config_.k < 1 || config_.k > knn_1_->getN() - 1)
    {
        KPROP_ERROR("invalid parameter k %d.\n", config_.k);
        exit(EXIT_FAILURE);
    }
    if (config_.variant == SWKProp || config_.variant == SWKPropPlus)
    {
        if (config_.alpha < 1.0F)
        {
            KPROP_ERROR("invalid amplifying factor %f.\n", config_.alpha);
            exit(EXIT_FAILURE);
        }
    }
    if (config_.variant == SWKPropPlus)
    {
        if (config_.rd <= 0.0F || config_.rd > 1.0F)
        {
            KPROP_ERROR("invalid refined dimension factor %f.\n", config_.rd);
            exit(EXIT_FAILURE);
        }
        if (config_.tc <= 0.0F || config_.tc > 1.0F)
        {
            KPROP_ERROR("invalid top-to-check factor %f.\n", config_.tc);
            exit(EXIT_FAILURE);
        }
    } 
    if (config_.beta <= 0.0F || config_.beta >= 1.0F)
    {
        KPROP_ERROR("invalid damping factor %f.\n", config_.beta);
        exit(EXIT_FAILURE);
    }
    if (config_.delta <= 0.0F)
    {
        KPROP_ERROR("invalid tolerance value %f.\n", config_.delta);
        exit(EXIT_FAILURE);
    }
    if (config_.max_iter < 1)
    {
        KPROP_ERROR("invalid max number of iterations %d.\n",
                    config_.max_iter);
        exit(EXIT_FAILURE);
    }

    /* compute max_knn_2 and min_knn_2 */
    /* they will be re-computed for SW-KProp+ */
    max_knn_2_ = 0.0F;
    min_knn_2_ = FLT_MAX;
    for (int i = 0; i < nr_data; i++)
    {
        for (int j = 1; j <= config_.k; j++)
        {
            float dist = (*knn_2_)(i,j);
            if (dist > max_knn_2_) max_knn_2_ = dist;
            if (dist < min_knn_2_) min_knn_2_ = dist;
        }
    }

    int ann_count = 0;
    for (int ii = 0; ii < (int) config_.ann_sizes->size(); ii++)
    {
        int i = (*(config_.ann_sizes))[ii];
        for (int j = 0; j < nr_exp; j++) /* j+1 is the experiment index */
        {
            string p_data_ann = config_.ped + config_.p_ann
                                + config_.dataset + "."
                                + T_as_string(i) + "_"
                                + T_as_string(j+1) + ".ann";
            /* pre-annotate data set */
            if (config_.verbose)
            {
                printf("  > Annotation %d/%d and Experiment %d/%d\n",
                       ann_count+1, nr_ann, (j+1), nr_exp);
                fflush(stdout);
            }
            annotate(p_data_ann);
            /* do kprop */
            kprop();
            compRecall  ((*kprop_recalls),    ann_count, j);
            compAccuracy((*kprop_accuracies), ann_count ,j);
            /* remove assignments, keep initial labeling
             * for next method (if any) */
            clearAssignments();
            /* end of kprop */

            /* rollback for next run, remove assignments and
             * initial labeling */
            rollback();
            /* flush output in each run */
            fflush(stdout);
        }

        /* code for result of each run can be written here */

        ann_count ++;
    }

    compAverage( *kprop_recalls, *kprop_accuracies );
    
    /* output results */
    if (config_.verbose)
    {
        printf("> Results:\n");
    }

    // printf("  # %s\n", KPROP_VAR[config_.variant]);
    // results(*kprop_recalls, *kprop_accuracies);
    resultsSim(*kprop_recalls, *kprop_accuracies);
    // printf("\n");
    delete kprop_recalls;
    delete kprop_accuracies;

}

void kprop::KProp::kprop()
{
    /* number of data items and distinct labels */
    int nr_data = (int) data_ -> size();
    int nr_labels = (int) labels_ -> size();

    /* here, we let "labeled" = "pre-annotated"
     * and "unlabeled" = "not pre-annotated"
     */
    const char *var = KPROP_VAR[config_.variant];
    if (config_.verbose)
    {
        printf("    > [%s] reordering data items...\n", var);
        fflush(stdout);
    }
    /* change the order of all data items --- let the first m labeled
     * and the rest unlabeled
     * oind --- old index
     * nind --- new index
     */
    int nr_labeled   = 0;
    for (int i = 0; i < (int)labels_->size(); ++i)
    {
        nr_labeled += (*labels_)[i].nr_annotated;
    }

    int nr_unlabeled = nr_data - nr_labeled;
    int oind_nind[nr_data]; /* old index to new index */
    int nind_oind[nr_data]; /* new index to old index */
    int labeled_count = 0;
    int unlabeled_count = 0;
    for (int i = 0; i < nr_data; i++)
    {
        if ((*data_)[i].annotated) /* a labeled data item */
        {
            oind_nind[i] = labeled_count;
            nind_oind[labeled_count++] = i;
        }
        else                       /* an unlabeled data item */
        {
            oind_nind[i] = nr_labeled + unlabeled_count;
            nind_oind[nr_labeled + unlabeled_count++] = i;
        }
    }
    if (config_.verbose)
    {
        printf("      #(labeled  data  items): %d\n", nr_labeled);
        printf("      #(unlabeled data items): %d\n", nr_unlabeled);
        fflush(stdout);
    }

    /* knn_1, knn_2, max_knn_2 and min_knn_2 are based on the original
     * knn matrices but might be changed (only in SW-KProp+) */
    DenMatSin* knn_1 = new DenMatSin(knn_1_->getM(), knn_1_->getN());
    DenMatSin* knn_2 = new DenMatSin(knn_2_->getM(), knn_2_->getN());
    float max_knn_2;
    float min_knn_2;
    knn_1->copy(*knn_1_);
    knn_2->copy(*knn_2_);
    max_knn_2 = max_knn_2_;
    min_knn_2 = min_knn_2_;

    if (config_.variant == SWKPropPlus) /* for SW-KProp+ only */
    {
        /* do feature selection */
        float rd = config_.rd;
        int   dim = descs_ -> getN();
        int   refined_dim = (int) (rd * dim);
              refined_dim = KPROP_MIN(dim, KPROP_MAX(refined_dim, 1));

        float tc = config_.tc;
        int   top = (int) (tc * nr_labeled);
              top = KPROP_MIN(nr_labeled-1, KPROP_MAX(top, 1));

        if (config_.verbose)
        {
            printf("    > [%s] selecting reduced feature sets...\n", var);
            printf("      refined dim: %d\n", refined_dim);
            printf("      top-to-check for discriminative ability: %d\n", top);
            fflush(stdout);
        }

        int nr_refined = 0;
        for (int i = 0; i < nr_labeled; i++)
        {
            int ii = nind_oind[i];
            int true_label = (*data_)[ii].label_index;

            /* compute the original discriminative ability */
            ifpair* sort_dist_0 = new ifpair[nr_labeled];
            for (int j = 0; j < nr_labeled; j++)
            {
                int jj = nind_oind[j];
                sort_dist_0[j].index = jj;
                if (jj == ii)
                    sort_dist_0[j].weight = 0.0F;
                else
                    sort_dist_0[j].weight = l2Dist(&((*descs_)(ii,0)),
                                                   &((*descs_)(jj,0)),
                                                   dim);
            }
            qsort(sort_dist_0, nr_labeled, sizeof(ifpair), cmp_ifpairs_i);
            int count_0 = 0; /* original discriminative ability */
            for (int j = 1; j <= top; j++)
            {
                if ( (*data_)[sort_dist_0[j].index].label_index
                    == true_label )
                    count_0 ++;
            }
            delete[] sort_dist_0;
            /***********/

            ifpair* sort_dim = new ifpair[dim];
            for (int d = 0; d < dim; d++) /* for each dim */
            {
                sort_dim[d].index = d;
                /* compute the discriminability for each dim */
                ifpair* sort_dist = new ifpair[nr_labeled];
                for (int j = 0; j < nr_labeled; j++)
                {
                    int jj = nind_oind[j];
                    sort_dist[j].index = jj;
                    if (jj == ii)
                    {
                        sort_dist[j].weight = 0.0F;
                    }
                    else
                    {
                        sort_dist[j].weight = l2Dist(&((*descs_)(ii,d)),
                                                     &((*descs_)(jj,d)),
                                                     1);
                    }
                }
                qsort(sort_dist, nr_labeled, sizeof(ifpair), cmp_ifpairs_i);
                int count = 0;
                for (int j = 1; j <= top; j++)
                {
                    if ( (*data_)[sort_dist[j].index].label_index
                            == true_label )
                        count ++;
                }
                sort_dim[d].weight = (float) count;
                delete[] sort_dist;
            }
            qsort(sort_dim, dim, sizeof(ifpair), cmp_ifpairs_d);
            int  size = refined_dim;
            int* feat = new int[size];
            for (int d = 0; d < size; d++)
            {
                feat[d] = sort_dim[d].index;
            }
            delete[] sort_dim;
            /* end of computing feature for p */

            /* compute the discriminative ability of the new feature */
            sort_dist_0 = new ifpair[nr_labeled];
            for (int j = 0; j < nr_labeled; j++)
            {
                int jj = nind_oind[j];
                sort_dist_0[j].index = jj;
                if (jj == ii)
                    sort_dist_0[j].weight = 0.0F;
                else
                    sort_dist_0[j].weight = l2DistPar(&((*descs_)(ii,0)),
                                                   &((*descs_)(jj,0)),
                                                   feat,
                                                   size);
            }
            qsort(sort_dist_0, nr_labeled, sizeof(ifpair), cmp_ifpairs_i);
            int count_1 = 0; /* discriminative ability of the new feature */
            for (int j = 1; j <= top; j++)
            {
                if ( (*data_)[sort_dist_0[j].index].label_index
                    == true_label )
                    count_1 ++;
            }
            delete[] sort_dist_0;
            /***********/

            if (count_1 <= count_0) /* no improvement */
                continue;

            nr_refined ++;

            /* use the computed feature to compute the new distances */
            /* labeled to all */
            ifpair* sort_dist = new ifpair[nr_data];
            for (int j = 0; j < nr_data; j++)
            {
                int jj = nind_oind[j];
                sort_dist[j].index  = jj;
                if (jj == ii)
                    sort_dist[j].weight = 0.0F;
                else
                    sort_dist[j].weight = l2DistPar(&((*descs_)(ii,0)),
                                                    &((*descs_)(jj,0)),
                                                    feat,
                                                    size);
            }
            qsort(sort_dist, nr_data, sizeof(ifpair), cmp_ifpairs_i);

            /* modify knn matrices */
            int K = knn_1->getN() - 1;
            for (int j = 1; j <= K; j++)
            {
                int jj = sort_dist[j].index;
                (*knn_1)(ii, j) = jj;
                (*knn_2)(ii, j) = sort_dist[j].weight;
            }

            delete[] sort_dist;
            delete[] feat;
        }
        /* re-compute max_knn_2 and min_knn_2 */
        max_knn_2 = 0.0F;
        min_knn_2 = FLT_MAX;
        for (int i = 0; i < nr_data; i++)
        {
            for (int j = 1; j <= config_.k; j++)
            {
                float dist = (*knn_2)(i,j);
                if (dist > max_knn_2) max_knn_2 = dist;
                if (dist < min_knn_2) min_knn_2 = dist;
            }
        }
        if (config_.verbose)
        {
            printf("      refined labeled: %f%%\n",
                    (float) nr_refined / nr_labeled * 100.0F);
        }
    }

    /* kprop algorithm */

    /* after reordering data items, propagation matrix P:
     *      P =  | P0  P1 |
     *           | P2  P3 |
     *      P0 is an identity matrix, P1 is a zero matrix.
     * score matrix S:
     *      S =  | S0 |
     *           | S1 |
     * iterations:
     *      S^q = P*S^{q-1}
     * Linear system: (I-P3)*S1 = P2S0
     * Let: X = S1 and H = P3 and B = P2S0
     *      X^q = HX^{q-1} + B
     * X^0 is a zero matrix. now let's construct H = P3 and P2 and S0
     */

    /* constructing P2 and P3
     * not optimal --- the symmetric property of P3 not used!
     * but already optimized for a good speed
     */
    if (config_.verbose)
    {
        printf("    > [%s] building influence graph with k = %d...\n",
                var, config_.k);
    }
    float alpha = config_.alpha; /* for SW-KProp and SW-KProp+ */

    /* a column of |P2 P3|, using columns is easy to track the reachability
     * of the influence graph */
    ifmap* cols = new ifmap[nr_data];
    /* sum of a row of |P2 P3| */
    float* sum_rows = new float[nr_unlabeled];
    /* initialize cols and sum_rows */
    for (int j = 0; j < nr_data; j++)      { cols[j].clear();    }
    for (int i = 0; i < nr_unlabeled; i++) { sum_rows[i] = 0.0F; }

    /* expanding the graph from source nodes */
    int item_level[nr_data]; /* level of items, using new index */
    int inf_level = 1000000; /* big enough for not connected */
    for (int i = 0; i < nr_data; i++)
    {
        if (i < nr_labeled)
            item_level[i] = 0;
        else
            item_level[i] = inf_level; /* initialize to infinite */
    }

    /* 0th level, labeled nodes,
     * for each labeled nodes (T), find top k unlabeled nodes (U), not
     * including labeled nodes themselves.
     * and then build the 1st level of links.
     */
    vector<int> curr_connected; /* new index */
    for (int j = 0; j < nr_labeled; j++) /* for each T, new index */
    {
        int jj = nind_oind[j]; /* old index */
        /* check kNN matrix */
        for (int k = 1; k <= config_.k; k++)
        {
            int ii = (*knn_1)(jj, k); /* old index */
            int i  = oind_nind[ii];    /* new index */

            if (item_level[i] == 0) /* a labeled node, skip */
                continue;

            float weight;
            switch (config_.variant)
            {
                case OriKProp:
                    weight = 1.0F;
                    break;
                case SWKProp:
                case SWKPropPlus:
                    weight = sim((*knn_2)(jj, k), max_knn_2, min_knn_2);
                    break;
                default:
                    /* should not be here */
                    KPROP_ERROR("undefined kprop variant.\n");
                    exit(EXIT_FAILURE);
                    break;
            }

            /* continue continue here */

            /* connect j and i */
            cols[j][i] = weight;
            if (item_level[i] == inf_level) /* an unlabeled node not seen
                                               before */
            {
                item_level[i] = 1;
                curr_connected.push_back(i);
            }
            /* else means item_level[i] == 1, an unlabeled node that has
             * already been connected by other T's */
            sum_rows[i - nr_labeled] += cols[j][i];
        }
    }
    /* other levels, from level 1 --- all unlabeled */
    int        curr_level = 1;
    int nr_curr_connected = (int) curr_connected.size();
    int      nr_connected = nr_curr_connected; /* only compute unlabeled */
    float temp1, temp2;

    while (nr_curr_connected > 0)
    {
        vector<int> newly_connected; /* nodes of next level */
        for (int c = 0; c < nr_curr_connected; c++) /* for each currently
                                                       connected U */
        {
            int j  = curr_connected[c]; /* new index */
            int jj = nind_oind[j];      /* old index */
            assert(item_level[j] == curr_level);

            for (int k = 1; k <= config_.k; k++)
            {
                int ii = (*knn_1)(jj, k); /* old index */
                int i  = oind_nind[ii];    /* new index */
                float weight;
                switch (config_.variant)
                {
                    case OriKProp:
                        weight = 1.0F;
                        break;
                    case SWKProp:
                    case SWKPropPlus:
                        weight = sim((*knn_2)(jj, k), max_knn_2, min_knn_2);
                        break;
                    default:
                        /* should not be here */
                        KPROP_ERROR("undefined kprop variant.\n");
                        exit(EXIT_FAILURE);
                        break;
                }
                
                /* check the level of i */
                if (item_level[i] == inf_level) /* an unlabeled node not
                                                   seen before */
                {
                    /* connect j and i, i and j */
                    cols[j][i] = weight;
                    cols[i][j] = weight;
                    /* change the level of this node */
                    item_level[i] = curr_level+1;
                    /* push this node to newly_connected */
                    newly_connected.push_back(i);
                    sum_rows[i - nr_labeled] += cols[j][i];
                    sum_rows[j - nr_labeled] += cols[i][j];
                }
                else if (item_level[i] == curr_level+1) /* an unlabeled
                                                           node connected by
                                                           someone in this
                                                           level */
                {
                    /* connect j and i, i and j */
                    cols[j][i] = weight;
                    cols[i][j] = weight;
                    sum_rows[i - nr_labeled] += cols[j][i];
                    sum_rows[j - nr_labeled] += cols[i][j];
                }
                else if (item_level[i] == curr_level) /* an unlabeled at
                                                         the same level */
                {
                    if (cols[i].find(j) == cols[i].end()) /* no <i,j> */
                    {
                        /* connect j and i, i and j */
                        cols[j][i] = weight;
                        cols[i][j] = weight;
                        sum_rows[i - nr_labeled] += cols[j][i];
                        sum_rows[j - nr_labeled] += cols[i][j];
                    }
                    else /* has <i,j> */
                    {
                        switch (config_.variant)
                        {
                            case OriKProp:
                                break;
                            case SWKProp:
                            case SWKPropPlus:
                                temp1 = cols[j][i];
                                temp2 = cols[i][j];
                                cols[j][i] = alpha * weight;
                                cols[i][j] = alpha * weight;
                                sum_rows[i - nr_labeled] += cols[j][i] - temp1;
                                sum_rows[j - nr_labeled] += cols[i][j] - temp2;
                                break;
                            default:
                                /* should not be here */
                                KPROP_ERROR("undefined kprop variant.\n");
                                exit(EXIT_FAILURE);
                                break;
                        }
                    }
                }
                else if (item_level[i] == 0) /* find a T */
                {
                    if (cols[i].find(j) == cols[i].end()) /* no <i,j> */
                    {
                        /* do nothing */
                    }
                    else /* has <i,j> */
                    {
                        switch (config_.variant)
                        {
                            case OriKProp:
                                break;
                            case SWKProp:
                            case SWKPropPlus:
                                temp2 = cols[i][j];
                                cols[i][j] = alpha * weight;
                                sum_rows[j - nr_labeled] += cols[i][j] - temp2;
                                break;
                            default:
                                /* should not be here */
                                KPROP_ERROR("undefined kprop variant.\n");
                                exit(EXIT_FAILURE);
                                break;
                        }
                    }
                }
                else if (item_level[i] == curr_level - 1) /* last level */
                {
                    if (cols[i].find(j) == cols[i].end()) /* no <i,j> */
                    {
                        cols[j][i] = weight;
                        cols[i][j] = weight;
                        sum_rows[i - nr_labeled] += cols[j][i];
                        sum_rows[j - nr_labeled] += cols[i][j];
                    }
                    else
                    {
                        switch (config_.variant)
                        {
                            case OriKProp:
                                break;
                            case SWKProp:
                            case SWKPropPlus:
                                temp1 = cols[j][i];
                                temp2 = cols[i][j];
                                cols[j][i] = alpha * weight;
                                cols[i][j] = alpha * weight;
                                sum_rows[i - nr_labeled] += cols[j][i] - temp1;
                                sum_rows[j - nr_labeled] += cols[i][j] - temp2;
                                break;
                            default:
                                /* should not be here */
                                KPROP_ERROR("undefined kprop variant.\n");
                                exit(EXIT_FAILURE);
                                break;
                        }
                    }
                }
                else /* other upper level */
                {
                    cols[j][i] = weight;
                    cols[i][j] = weight;
                    sum_rows[i - nr_labeled] += cols[j][i];
                    sum_rows[j - nr_labeled] += cols[i][j];
                }
            }
        } /* end for (int c = 0; c < nr_curr_connected; c++) */

        curr_level++;
        curr_connected = newly_connected;
        nr_curr_connected = (int) curr_connected.size();
        nr_connected += nr_curr_connected;
    } /* end while (nr_curr_connected > 0) */

    /* connect the rest */
    /* all nodes not expanded will be referred as one single level ---
     * the max level */
    int max_level = curr_level;
    // int nr_levels = max_level + 1;
    for (int j = 0; j < nr_data; j++)
    {
        if (item_level[j] != inf_level)
            continue;

        item_level[j] = max_level;
        int jj = nind_oind[j];
        for (int k = 1; k <= config_.k; k++)
        {
            int ii = (*knn_1)(jj, k); /* old index */
            int i  = oind_nind[ii];   /* new index */
            float weight;
            switch (config_.variant)
            {
                case OriKProp:
                    weight = 1.0F;
                    break;
                case SWKProp:
                case SWKPropPlus:
                    weight = sim((*knn_2)(jj, k), max_knn_2, min_knn_2);
                    break;
                default:
                    /* should not be here */
                    KPROP_ERROR("undefined kprop variant.\n");
                    exit(EXIT_FAILURE);
                    break;
            }
            /* check the level of i */
            if (item_level[i] == inf_level)
            {
                /* connect j and i, i and j */
                cols[j][i] = weight;
                cols[i][j] = weight;
                sum_rows[i - nr_labeled] += cols[j][i];
                sum_rows[j - nr_labeled] += cols[i][j];
            }
            else if (item_level[i] == max_level)
            {
                if (cols[i].find(j) == cols[i].end()) /* no <i,j>*/
                {
                    /* connect j and i, i and j */
                    cols[j][i] = weight;
                    cols[i][j] = weight;
                    sum_rows[i - nr_labeled] += cols[j][i];
                    sum_rows[j - nr_labeled] += cols[i][j];
                }
                else
                {
                    switch (config_.variant)
                    {
                        case OriKProp:
                            break;
                        case SWKProp:
                        case SWKPropPlus:
                            temp1 = cols[j][i];
                            temp2 = cols[i][j];
                            cols[j][i] = alpha * weight;
                            cols[i][j] = alpha * weight;
                            sum_rows[i - nr_labeled] += cols[j][i] - temp1;
                            sum_rows[j - nr_labeled] += cols[i][j] - temp2;
                            break;
                        default:
                            /* should not be here */
                            KPROP_ERROR("undefined kprop variant.\n");
                            exit(EXIT_FAILURE);
                            break;
                    }
                }
            }
            else if (item_level[i] == 0)
            {
                /* do nothing */
            }
            else /* upper level except than 0th level */
            {
                /* connect j and i, i and j */
                cols[j][i] = weight;
                cols[i][j] = weight;
                sum_rows[i - nr_labeled] += cols[j][i];
                sum_rows[j - nr_labeled] += cols[i][j];
            }
        }
    }
    delete knn_1;
    delete knn_2;

    //  /***********debug************/
    //  /* this is slow */
    //  /* check sum_rows */
    //  float* sum_rows2 = new float[nr_unlabeled];
    //  for (int i = 0; i < nr_unlabeled; i++) { sum_rows2[i] = 0.0F; }
    //  // for (int i = 0; i < nr_unlabeled; i++) { sum_rows[i] = 0.0F; }

    //  for (int i = nr_labeled; i < nr_data; i++)
    //  {
    //      for (int j = 0; j < nr_data; j++)
    //      {
    //          if (cols[j].find(i) != cols[j].end())
    //          {
    //              sum_rows2[i - nr_labeled] += cols[j][i];
    //              // sum_rows[i - nr_labeled] += cols[j][i];
    //          }
    //      }
    //  }
    //  float lm = 0;
    //  for (int i = 0; i < nr_unlabeled; i++)
    //  {
    //      float m = (float) fabs(sum_rows2[i]-sum_rows[i]);
    //      if (m > lm)
    //          lm = m;
    //  }
    //  printf("%f\n", lm);
    //  
    //  assert(0);
    //  /*****end of debug***********/
    
    if (config_.verbose)
    {
        printf("      Expanded unlabeled nodes: %d / %d = %.2f%%\n",
                nr_connected, nr_unlabeled, 
                (float)nr_connected/nr_unlabeled*100.0F);
    }

    /* construct two sparse matrix P2 and P3 */
    ifmap_cit cit;
    FSp_mat* P2 = new FSp_mat(nr_unlabeled, nr_labeled);
    FSp_mat* P3 = new FSp_mat(nr_unlabeled, nr_unlabeled);
    for (int j = 0; j < nr_data; j++)
    {
        for (cit = cols[j].begin(); cit != cols[j].end(); cit++)
        {
            int i = cit->first - nr_labeled;
            float weight = (sum_rows[i] < TOLERANCE_VALUE) ?
                            cit->second :
                            cit->second / sum_rows[i] * config_.beta;
            if (j < nr_labeled)
            {
                    P2->insert_entry(weight, i, j);
            }
            else
            {
                    P3->insert_entry(weight, i, j - nr_labeled);
            }
        }
    }
    delete[] sum_rows;
    delete[] cols;

    P2->end_construction();
    P3->end_construction();
    /* end of construction, P2 and P3 are ready */
    /* compute S0 */
    DenMatSin *S0 = new DenMatSin(nr_labeled, nr_labels);
    for (int i = 0; i < nr_labeled; i++)
    {
        int ii = nind_oind[i];
        (*S0)(i, (*data_)[ii].label_index) = 1.0F;
    }
    /* compute B = P2S0 */
    DenMatSin* B = new DenMatSin(nr_unlabeled, nr_labels);
    MatMultiply2(*P2, *S0, *B);
    delete P2;
    delete S0;
    if (config_.verbose)
    {
        printf("    > [%s] propagating influence scores...\n", var);
        fflush(stdout);
    }
    /* X = S1, initially all zeros */
    /* score matrix X for unlabeled data */
    DenMatSin* X = new DenMatSin(nr_unlabeled, nr_labels);
    // bool converged = iterate(X, P3, B);
    iterate(X, P3, B);
    
    delete B;
    delete P3;

    if (config_.verbose)
    {
        printf("    > [%s] assigning labels to unlabeled data items...\n",
                var);
        fflush(stdout);
    }

    int nr_zero_score_assignments = 0;
    float max_score;
    float max_index;
    for (int i = 0; i < nr_unlabeled; i++)
    {
        int ii = nind_oind[i+nr_labeled];
        max_score = (*X)(i,0);
        max_index = 0;
        for (int j = 1; j < nr_labels; j++)
        {
            if ( (*X)(i,j) > max_score)
            {
                max_score = (*X)(i,j);
                max_index = j;
            }
        }
        if (max_score < TOLERANCE_VALUE) nr_zero_score_assignments ++;
        (*data_)[ii].assignment = max_index;
        if ( (*data_)[ii].label_index == max_index )
        {
            (*labels_)[max_index].nr_correct_assignments ++;
        }
    }
    if (config_.verbose)
    {
        printf("      #(0-score assignments) / #(assignments) = %d / %d ="
               "%.2f%%\n",
               nr_zero_score_assignments, nr_unlabeled,
               (float)nr_zero_score_assignments/nr_unlabeled*100.0F);
        fflush(stdout);
    }
    delete X;
}

bool kprop::KProp::iterate(DenMatSin* X, FSp_mat* P3, DenMatSin* B)
{
    int row = X->getM();
    int col = X->getN();
    /* X2 is used to store intermediate result of X */
    DenMatSin* X2 = new DenMatSin(row, col);
    /* X = P3*X + B */
    int numbers_per_line = 10;
    int iter;
    bool converged = false;
    for (iter = 1; iter <= config_.max_iter; iter++)
    {
        if (config_.verbose)
        {
            if ( (iter-1)%numbers_per_line== 0 )
                printf("      ");
            printf("%d%c", iter, 
                   (iter==config_.max_iter||
                   (iter-numbers_per_line)%numbers_per_line==0)?'\n':' ');
            fflush(stdout);
        }
        if (iter % 2 == 1) /* odd number */
        {
            // X2 = P3*X + B
            MatMultiply2(*P3, *X, *X2);
            MatAdd(*X2, *B, *X2);
        }
        else /* even */
        {
            // X = P3*X2 + B
            MatMultiply2(*P3, *X2, *X);
            MatAdd(*X, *B, *X);
        }
        converged = true;
        for (int i = 0 ; i < row; ++i)
        {
            for (int j = 0; j < col; ++j) 
            {
                if (fabs((*X)(i,j)-(*X2)(i,j)) > (double)config_.delta)
                {
                    converged = false;
                    break;
                }
            }
            if (!converged) break;
        }
        if (converged) break;
    }

    if (config_.max_iter % 2 == 1)
    {
        X->copy(*X2);
    }
    delete X2;
    if (config_.verbose)
    {
        if ( (iter-numbers_per_line) % numbers_per_line != 0 )
        {
            printf("\n");
        }
        printf("      %s\n", converged?"converged.":"not converged.");
        fflush(stdout);
    }
    return converged;
}

void kprop::KProp::compRecall(DenMatSin& recalls, int i, int j)
{
    if (config_.verbose)
    {
        printf("    > Computing recall...\n");
        fflush(stdout);
    }
    int nr_labels              = (int)labels_->size();
    int nr_unlabeled_data      = 0;
    int nr_correct_assignments = 0;
    for (int k = 0; k < nr_labels; k++)
    {
        nr_unlabeled_data      += (*labels_)[k].data_indices->size()
                                  - (*labels_)[k].nr_annotated;
        nr_correct_assignments += (*labels_)[k].nr_correct_assignments;
    }
    float recall = (float) nr_correct_assignments / nr_unlabeled_data;
    if (config_.verbose)
    {
        printf("      #(correct  assignments): %d\n",
                                                   nr_correct_assignments);
        printf("      #(unlabeled data items): %d\n", nr_unlabeled_data);
        printf("      recall = %.2f%%\n", recall*100.0F);
        fflush(stdout);
    }
    recalls(i,j) = recall;
}
void kprop::KProp::compAccuracy(DenMatSin& accuracies, int i, int j)
{
    if (config_.verbose)
    {
        printf("    > Computing accuracies...\n");
        fflush(stdout);
    }
    int   nr_labels = (int)labels_->size();
    float accuracy;
    int   numbers_per_line = 5;
    for (int k = 0; k < nr_labels; k++)
    {
        accuracy = ( (float) (*labels_)[k].nr_correct_assignments )
                     / ( (float) (*labels_)[k].data_indices->size()
                                - (*labels_)[k].nr_annotated);
        accuracies(i, j*nr_labels + k) = accuracy;
        if (config_.verbose)
        {
            if (k % numbers_per_line == 0)
                printf("      ");
            printf("%6.2f%%%c", accuracy * 100.0F, ((k==nr_labels-1)
                   ||(k+1-numbers_per_line)%numbers_per_line==0)?'\n':' ');
        }
    }
}
void kprop::KProp::compAverage(DenMatSin& recalls, DenMatSin& accuracies)
{
    int nr_ann = recalls.getM();
    int nr_exp = recalls.getN() - 1;
    /* compute average recall */
    for (int i = 0; i < nr_ann; i++)
    {
        for (int j = 0; j < nr_exp; j++)
        {
            recalls(i, nr_exp) += recalls(i, j);
        }
        recalls(i, nr_exp) /= (float)nr_exp;
    }
    /* compute average accuracy and sem (standard error of mean)
     * sem = std / sqrt(N) = 1/(N(N-1)) * sum( (x_i-x_bar)^2 )
     */
    nr_ann = accuracies.getM();
    int N  = accuracies.getN() - 2;
    float mean; /* mean */
    float diff; /* temp difference xi-x_bar */
    float sum;  /* sum of (xi-x_bar)^2 */
    float sem;  /* sem */
    for (int i = 0; i < nr_ann; i++)
    {
        mean = 0.0F;
        sum = 0.0F;
        for (int j = 0; j < N; j++)
        {
            mean += accuracies(i, j);
        }
        mean /= (float)N;
        accuracies(i, N) = mean;
        for (int j = 0; j < N; j++)
        {
            diff = accuracies(i,j) - mean;
            sum += diff * diff;
        }
        sem = (float) sqrt ( 1.0F / N / (N-1) * sum);
        accuracies(i, N+1) = sem;
    }
}

void kprop::KProp::results( const DenMatSin& recalls,
		            const DenMatSin& accuracies)
{
    printf("  # Lb sz/file Average_recall%% Average_accuracy%% "
           "Standard_error_mean%%\n");
    int nr_exp    = recalls.getN() - 1;
    int N         = accuracies.getN() - 2;
    int ann_count = 0;
    /* avg of avg recall */
    float avg_recall = 0.0F;

    for (int ii = 0; ii < (int) config_.ann_sizes->size(); ii++)
    {
        int i = (*(config_.ann_sizes))[ii];
        printf( "    %10d %14.2f%% %16.2f%% %19.2f%%\n",
                i,
                recalls(ann_count, nr_exp)*100.0F,
                accuracies(ann_count, N)*100.0F,
                accuracies(ann_count, N+1)*100.0F);

        avg_recall += recalls(ann_count, nr_exp)*100.0F;
        ann_count++;
    }
    avg_recall /= (float) config_.ann_sizes->size();
    printf("  # Average of Average recall: %.2f%%\n", avg_recall);
}

void kprop::KProp::resultsSim( const DenMatSin& recalls,
		            const DenMatSin& accuracies)
{
    // printf("  # Lb sz/file Average_recall%% Average_accuracy%% "
    //        "Standard_error_mean%%\n");
    int nr_exp    = recalls.getN() - 1;
    // int N         = accuracies.getN() - 2;
    int ann_count = 0;
    /* avg of avg recall */
    float avg_recall = 0.0F;

    for (int ii = 0; ii < (int) config_.ann_sizes->size(); ii++)
    {
        // int i = (*(config_.ann_sizes))[ii];
        // printf( "    %10d %14.2f%% %16.2f%% %19.2f%%\n",
        //         i,
        //         recalls(ann_count, nr_exp)*100.0F,
        //         accuracies(ann_count, N)*100.0F,
        //         accuracies(ann_count, N+1)*100.0F);

        avg_recall += recalls(ann_count, nr_exp)*100.0F;
        ann_count++;
    }
    avg_recall /= (float) config_.ann_sizes->size();
    // printf("  # Average of Average recall: %.2f%%\n", avg_recall);
    printf("%.2f%% ", avg_recall);
}

void kprop::KProp::rollback()
{
    for (int i = 0; i < (int) data_->size(); i++)
    {
        (*data_)[i].annotated  = false;
        (*data_)[i].assignment = -1;
    }
    for (int i = 0; i < (int) labels_->size(); i++)
    {
        (*labels_)[i].nr_annotated           = 0;
        (*labels_)[i].nr_correct_assignments = 0;
    }
}

void kprop::KProp::clearAssignments()
{
    for (int i = 0; i < (int) data_->size(); i++)
    {
        (*data_)[i].assignment = -1;
    }
    for (int i = 0; i < (int) labels_->size(); i++)
    {
        (*labels_)[i].nr_correct_assignments = 0;
    }
}
