/* Derived from the modified nndes-common.h as of June 18, 2013 see comments
 * below
 *
 * Jichao Sun (js87@njit.edu)
 *
 *   July 12, 2013
 *     whether or not OpenMP is totally controled by the new Makefile
 *     if "omp.h" is not loaded update() and update_unsafe() will be roughly
 *     the same (except than the return value)
 *   July 11, 2013
 *     added another recall computation function
 *     float recall (const KNN &knn, const KNN &ans, int K)
 */

/* modified by Jichao Sun on June 18, 2013
 *  added float recall(const float* knn, const KNN&ans, int K)
 */

/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#ifndef WDONG_NNDESCENT_COMMON
#define WDONG_NNDESCENT_COMMON

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <boost/assert.hpp>
#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace nndes 
{

    using std::vector;
    using std::numeric_limits;

#define SYMMETRIC 1

#ifdef _OPENMP
#if SYMMETRIC
#define NEED_LOCK 1
#endif
#endif

#if NEED_LOCK
    class Mutex 
    {
        omp_lock_t *lock;
    public:
        Mutex (): lock(0) { }

        void init () 
        {
            lock = new omp_lock_t;
            omp_init_lock(lock);
        }

        ~Mutex () 
        {
            if (lock)
            {
                omp_destroy_lock(lock);
                delete lock;
            }
        }

        void set () 
        {
            omp_set_lock(lock);
        }

        void unset () 
        {
            omp_unset_lock(lock);
        }

        friend class ScopedLock;
    };

    /* not used. ~Jichao */
    class ScopedLock 
    {
        omp_lock_t *lock;
    public:
        ScopedLock (Mutex &mutex) 
        {
            lock = mutex.lock;
            omp_set_lock(lock);
        }
        ~ScopedLock () 
        {
            omp_unset_lock(lock);
        }
    };
#else
    class Mutex 
    {
    public:
        void init () {};
        void set () {};
        void unset () {};
    };
    class ScopedLock 
    {
    public:
        ScopedLock (Mutex &) { }
    };
#endif

    struct KNNEntry
    {
        static const int BAD = -1; //numeric_limits<int>::max();
        int key;
        float dist;   
        bool flag;
        bool match (const KNNEntry &e) const { return key == e.key; }
        KNNEntry (int key_, float dist_, bool flag_ = true)
            :key(key_), dist(dist_), flag(flag_) { }
        KNNEntry () : dist(numeric_limits<float>::max()) { }
        void reset () { key = BAD;  dist = numeric_limits<float>::max(); }
        friend bool operator < (const KNNEntry &e1, const KNNEntry &e2)
        {
            return e1.dist < e2.dist;
        }
    };

    class KNN: public vector<KNNEntry>
    {
        int K;
        Mutex mutex;
    public:
        typedef KNNEntry Element;
        typedef vector<KNNEntry> Base;

        void init (int k) 
        {
            mutex.init();
            K = k;
            this->resize(k);
            BOOST_FOREACH(KNNEntry &e, *this) {
                e.reset();
            }
        }

        int update (Element t)
        {
            //ScopedLock ll(mutex);
            mutex.set();
            int i = this->size() - 1;
            int j;
            if (!(t < this->back())) 
            {
                mutex.unset();
                return -1;
            }
            for (;;)
            {
                if (i == 0) break;
                j = i - 1;
                if (this->at(j).match(t)) 
                {
                    mutex.unset();
                    return -1;
                }
                if (this->at(j) < t) break;
                i = j;
            }

            j = this->size() - 1;
            for (;;)
            {
                if (j == i) break;
                this->at(j) = this->at(j-1);
                --j;
            }
            this->at(i) = t;
            mutex.unset();
            return i;
        }

        void update_unsafe (Element t)
        {
            int i = this->size() - 1;
            int j;
            if (!(t < this->back())) return;
            for (;;)
            {
                if (i == 0) break;
                j = i - 1;
                if (this->at(j).match(t)) 
                {
                    return;
                }
                if (this->at(j) < t) break;
                i = j;
            }

            j = this->size() - 1;
            for (;;)
            {
                if (j == i) break;
                this->at(j) = this->at(j-1);
                --j;
            }
            this->at(i) = t;
        }

        void lock() 
        {
            mutex.set();
        }

        void unlock() 
        {
            mutex.unset();
        }
    };

    static inline float recall (const int *knn, const KNN &ans, int K) 
    {
        int match = 0;
        for (int i = 0; i < K; ++i) 
        {
            for (int j = 0; j < K; ++j) 
            {
                if (knn[i] == ans[j].key) 
                {
                    ++match;
                    break;
                }
            }
        }
        return float(match) / K;
    }

    static inline float recall (const float *knn, const KNN &ans, int K) 
    {
        int match = 0;
        for (int i = 0; i < K; ++i) 
        {
            for (int j = 0; j < K; ++j) 
            {
                if (int(knn[i]) == ans[j].key) 
                {
                    ++match;
                    break;
                }
            }
        }
        return float(match) / K;
    }

    static inline float recall (const KNN &knn, const KNN &ans, int K) 
    {
        int match = 0;
        for (int i = 0; i < K; ++i) 
        {
            for (int j = 0; j < K; ++j) 
            {
                if (knn[i].key == ans[j].key) 
                {
                    ++match;
                    break;
                }
            }
        }
        return float(match) / K;
    }


    class Random {
        boost::mt19937 rng;
    public:
        Random () { }
        void seed (int s) { rng.seed(s); }
        ptrdiff_t operator () (ptrdiff_t i) { return rng() % i; }
    };
}

#endif 

