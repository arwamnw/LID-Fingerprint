/* KProp.h */

/* KProp class
 * Jichao Sun (js87@njit.edu)
 * 
 * June 12, 2013
 *   added iterative() to compute X=P3*X+B
 *
 * June 7, 2013
 *   added sim() for the similarity function of SW-KProp and SW-KProp+
 *
 * June 6, 2013 Parameter names changed to be consistent with the paper
 *     df --> beta
 *     af --> alpha
 *   Bestmatch method(s) removed 
 *
 * June 5, 2013 Initialized based on KProp.h (last modified on Mar 14,
 * 2013) from kprop.6
 *   KProp variants changed
 *
 */

#ifndef KPROP_KPROP_H_
#define KPROP_KPROP_H_

#include <string>
using std::string; 
#include <vector>
using std::vector;

#include "global.h"
#include "DenMatSin.h"
#include "nist_spblas/nist_spblas.h"
using namespace NIST_SPBLAS;


namespace kprop
{

/* kprop variants */
typedef enum KPropVar_
{
    OriKProp    = 0, /* KProp */
    SWKProp     = 1, /* SW-KProp */
    SWKPropPlus = 2, /* SW-KProp+ */
} KPropVar;
/* names of kprop variants */
extern const char* KPROP_VAR[];
/* kprop configuration structure */
typedef struct KPropConf_
{
    /* system configs */
    bool     verbose;   /* verbose output */
    KPropVar variant;   /* 0: KProp, 1: SW-KProp, 2: SW-KProp+ */
    string   pwd;       /* working directory */
    string   dataset;   /* dataset name */
    string   ped;       /* experiment directory */
    string   p_ann;     /* (relative) path of annotation directory */
    bool     run;       /* whether to run kprop or to annotate */
    vector<int>* ann_sizes; /* annotation sizes: numbers of data per
                               category to prelabel */
    int      nr_exp;    /* number of experiments per ann size */
    /* KProp parameters */
    int      k;         /* parameter k (kNN) of the influence graph */
    float    beta;      /* parameter beta (damping factor) */
    /* SW-KProp parameters */
    float    alpha;     /* amplifying factor */
    /* SW-KProp+ parameters */
    float    rd;        /* refined dimension factor */
    float    tc;        /* top-to-check factor */
    /* other parameters */
    float    delta;     /* tolerance value for convergence test */
    int      max_iter;  /* max number of iterations */

} KPropConf;

/* Classification KProp */
class KProp
{
    public:
        KProp();                     /* constructor, initializes member
                                        variables */
        KProp(const KPropConf& cfg); /* constructor, initializes member
                                        variables and configurations */
        ~KProp();                    /* destructor */
        void start();                /* start annotate() or run() */

        void start(DenMatSin* knn_1, DenMatSin* knn_2); /*direct start */
        

    private:
        /* data types */
        /* a single data item (an "object") */
        typedef struct Datum_
        {
            string key;         /* a unique identifier */
            int    label_index;	/* index of the associated ground truth
                                   label */
            bool   annotated;	/* whether pre-annotated */
            int    assignment;	/* index of assigned label */
        } Datum;
        /* a unique text label */
        typedef struct Label_
        {
            string       text;          /* label text */
            vector<int>* data_indices;  /* indices of data items associated
                                           with this label */
            int          nr_annotated;  /* number of data pre-annotated by
                                           this label */
            int          nr_correct_assignments;
                                        /* number of data associated with
                                         * this label that are finally
                                         * correctly assigned
                                         * excluding pre-annotated data
                                         */
        } Label;

        /* core data */
        KPropConf      config_;      /* configurations */
        vector<Datum>* data_;        /* data items */
        vector<Label>* labels_;      /* unique labels */
        simap*         key_index_;   /* data item ---> data index */
        simap*         label_index_; /* label ---> label index */

        DenMatSin*     descs_;       /* descriptors of all data items */
        DenMatSin*     knn_1_;       /* kNN matrix of data items */
        DenMatSin*     knn_2_;       /* distance matrix corresponding to
                                        knn_1_ */
        float          max_knn_2_;   /* max value of knn_2_ (the 1st to
                                        the config_.k-th columns used; the 0th
                                        column is 0.0F for the distance
                                        to the data item itself */
        float          min_knn_2_;   /* min value of knn_2_ (the 1st to
                                        the config_.k-th columns used) */
        
        /* functions */
        /* level 0 routines */
        /* initializes the member variables */
        void init();

        /* level 1 routines */
        /* initializes data items */
        void initData();
        /* initializes ground truth labels */
        void initLabels();

        /* level 2 routines */
        /* pre-annotate dataset */
        void annotate();
        /* pre-annotates data items according to the ann (label) size:
         * number of data per category to prelabel,
         * returns the total number of data items pre-annotated
         */
        int annotate(int ann_size);
        /* pre-annotates data items according to annotation file
         * returns the total number of data items pre-annotated
         */
        int annotate(const string& ann_file);
        /* pre-annotates one data item, called by the above two functions */
        void annotateOne(int data_index);

        /* level 3 routines (methods that compute propagation scores
         * and do assignment
         */
        void run();                  /* run algorithm(s) */
        /* KProp, SW-KProp or SW-KProp+ method */
        void kprop();
        /* similarity function used in SW-KProp and SW-KProp+ */
        inline float sim(float d, float dmax, float dmin)
        {
            float s = 1.0F - (d - dmin) / (dmax - dmin);
            return s*s;
        }
        /* iteratively compute X = P3*X + B, returns if converged */
        bool iterate(DenMatSin* X, FSp_mat* P3, DenMatSin* B);


        /* other routines */
        /* computes the recall of a specific ann size and a specific
         * experiment and fills in the recall matrix
         */
        void compRecall(DenMatSin& recalls, int i, int j);
        /* computes the accuracy of a specific ann size and a specific
         * experiment and fills in the accuracy matrix
         */
        void compAccuracy(DenMatSin& accuracies, int i, int j);
        /* computes the average recall and the average accuracy and
         * the standard error mean
         */
        void compAverage(DenMatSin& recalls, DenMatSin& accuracies);
        /* display results */
        void results(const DenMatSin& recalls, const DenMatSin& accuracies);
        void resultsSim(const DenMatSin& recalls, const DenMatSin& accuracies);
        /* rolls back to post-level-1 (pre-pre-annotating) status */
        void rollback();
        /* rolls back to post-level-2 (clears assignments but keeps
         * pre-annotating information
         */
        void clearAssignments();
}; /* class KProp */

} /* namespace kprop */

#endif /* KPROP_KPROP_H_ */
