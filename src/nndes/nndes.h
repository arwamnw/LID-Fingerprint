/* Derived from the original nndes.h
 *
 * Jichao Sun (js87@njit.edu)
 *
 * Oct 3, 2013
 *
 * July 23, 2013
 *   added some stuff for NNF-Descent
 * 
 * July 17, 2013
 *   TEST_CODE is not defined here, use c++ compile flag -DTEST_CODE
 *
 *
 *   July 12, 2013
 *     #pragma is conditional, controled by the new Makefile
 *     added some test code (a bug fix), undef TEST_CODE to recover to original
 */

/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#ifndef WDONG_NNDESCENT
#define WDONG_NNDESCENT

#include <cassert>
#include "nndes-common.h"
#include "../DenMatSin.h"
using kprop::DenMatSin;


namespace nndes 
{
    /* To re-sort KNN entries in an KNN list in ascending order
     * This function can be moved to nndes-common.h
     * Will have function pointer problem if moved into NNDescent class
     * as a member function
     */
    bool compKnnEntries(const KNN::Element& e1, const KNN::Element& e2)
    {
        // return (e1.dist < e2.dist);
        return (e1 < e2);
    }

    using std::cerr;
    using std::vector;
    using std::swap;
    using boost::progress_display;
    using std::sort;

// #ifndef NNDES_SHOW_PROGRESS
// #define NNDES_SHOW_PROGRESS 1
// #endif

    // Normally one would use GRAPH_BOTH,
    // GRAPH_KNN & GRAPH_RNN are for experiments only.
    static const int GRAPH_NONE = 0, GRAPH_KNN = 1, GRAPH_RNN = 2, GRAPH_BOTH = 4;
    typedef int GraphOption;

    // The main NN-Descent class.
    // Instead of the actual dataset, the class takes a distance oracle
    // as input.  Given two data item ids, the oracle returns the distance
    // between the two.
    template <typename ORACLE>
    class NNDescent 
    {
    private:
        const ORACLE &oracle;
        int N;              // # points
        int K;              // K-NN to find
        int S;              // # of NNs to use for exploration
                            // note: this is originally used in NN-Descent,
                            // we use this in NNF-Descent,
                            // NN-Descent always use S=K;
        GraphOption option;
        vector<KNN> nn;     // K-NN approximation

        // We maintain old and newly added KNN/RNN items
        // separately for incremental processing:
        // we need to compare two new ones
        // and a new one to an old one, but not two old ones as they
        // must have been compared already.
        vector<vector<int> > nn_old;
        vector<vector<int> > nn_new;
        vector<vector<int> > rnn_old;
        vector<vector<int> > rnn_new;



        // total number of comparisons done.
        long long int cost;

        /* for NNF-Descent, we need to save an active full size KNN and RNN */
        vector<vector<int> > nn_full;
        vector<vector<int> > rnn_full;

        // This function decides of it's necessary to compare two
        // points.  Obviously a point should not compare against itself.
        // Another potential usage of this function is to record all
        // pairs that have already be compared, so that when seen in the future,
        // then same pair doesn't have be compared again.
        bool mark (int p1, int p2) { return p1 == p2; }

        // Compare two points and update their KNN list of necessary.
        // Return the number of comparisons done (0 or 1).
        int update (int p1, int p2) 
        {
            if (mark(p1, p2)) return 0;
            // KNN::update is synchronized by a lock
            // keep an order is necessary to avoid deadlock.
            if (p1 > p2) swap(p1, p2);
            float dist =  oracle(p1, p2);
            nn[p1].update(KNN::Element(p2, dist, true));
            nn[p2].update(KNN::Element(p1, dist, true));
            return 1;
        }


    public:
        const vector<KNN> &getNN() const { return nn; }

        long long int getCost () const { return cost; }

        // NNDescent (int N_, int K_, float S_, const ORACLE &oracle_,
        //            GraphOption opt = GRAPH_BOTH)
        //     : oracle(oracle_), N(N_), K(K_), S(K * S_), option(opt), nn(N_),
        //       nn_old(N_), nn_new(N_), rnn_old(N_), rnn_new(N_), cost(0)
        NNDescent (int N_, int K_, float S_, const ORACLE &oracle_,
                   GraphOption opt = GRAPH_BOTH)
            : oracle(oracle_), N(N_), K(K_), S(K * S_), option(opt), nn(N_),
              nn_old(N_), nn_new(N_), rnn_old(N_), rnn_new(N_), cost(0),
              nn_full(N_), rnn_full(N_)
        {
            for (int i = 0; i < N; ++i) 
            {
                nn[i].init(K);
                // random initial edges
                if ((option & GRAPH_KNN) || (option & GRAPH_BOTH)) 
                {
                    // nn_new[i].resize(S);
                    nn_new[i].resize(K);
                    BOOST_FOREACH(int &u, nn_new[i]) 
                    {
                        u = rand() % N;
                    }
                }
                if ((option & GRAPH_RNN) || (option & GRAPH_BOTH)) 
                {
                    // rnn_new[i].resize(S);
                    rnn_new[i].resize(K);
                    BOOST_FOREACH(int &u, rnn_new[i]) 
                    {
                        u = rand() % N;
                    }
                }

                /* random nn_full and rnn_full in case that the initial
                 * NN-Descent is not performed at all and we still have
                 * a valid random knn graph */
                nn_full[i].resize(K);
                for (int k = 0; k < K; ++k)
                {
                    bool ok;
                    int u;
                    do
                    {
                        u = rand() % N;
                        ok = true;
                        if (u == i)
                        {
                            ok = false;
                            continue;
                        }
                        for (int k2 = 0; k2 < k; ++k2)
                        {
                            if (u == nn_full[i][k2])
                            {
                                ok = false;
                                break;
                            }
                        }
                    } while (!ok);
                    nn_full[i][k] = u;
                    rnn_full[u].push_back(i);
                }
            }
        }

        /* build nn directly from nn_full */
        void directNN(void)
        {
            for (int n = 0; n < N; ++n)
            {
                for (int k = 0; k < K; ++k)
                {
                    nn[n][k].key = nn_full[n][k];
                    nn[n][k].dist = oracle(n, nn[n][k].key);
                    nn[n][k].flag = true;
                }
            }
        }

        // An iteration contains two parts:
        //      local join
        //      identify the newly detected NNs.
        int iterate () 
        {
#if NNDES_SHOW_PROGRESS
            progress_display progress(N, cerr);
#endif

            long long int cc = 0;
            // local joins
#ifdef _OPENMP
#pragma omp parallel for default(shared) reduction(+:cc)
#endif
            for (int i = 0; i < N; ++i) 
            {
                // The following loops are bloated to deal with all
                // the experimental setups.  Otherwise they should
                // be really simple.

                /* I think the original case, when option == GRAPH_BOTH,
                 * the first two cases are ignored... (as KNN and RNN
                 * seperately), a quick fix is in the TEST_CODE section.
                 * ~Jichao */
#ifndef TEST_CODE
                if (option & GRAPH_KNN)
#else
                if ( (option & GRAPH_KNN) || (option & GRAPH_BOTH) )
#endif
                {
                    BOOST_FOREACH(int j, nn_new[i]) 
                    {
                        BOOST_FOREACH(int k, nn_new[i]) 
                        {
                            if (j >= k) continue;
                            cc += update(j, k);
                        }
                        BOOST_FOREACH(int k, nn_old[i]) 
                        {
                            cc += update(j, k);
                        }
                    }
                }

#ifndef TEST_CODE
                if (option & GRAPH_RNN) 
#else
                if ( (option & GRAPH_RNN) || (option & GRAPH_BOTH) )
#endif
                {
                    BOOST_FOREACH(int j, rnn_new[i]) 
                    {
                        BOOST_FOREACH(int k, rnn_new[i]) 
                        {
                            if (j >= k) continue;
                            cc += update(j, k);
                        }
                        BOOST_FOREACH(int k, rnn_old[i]) 
                        {
                            cc += update(j, k);
                        }
                    }
                }

                if (option & GRAPH_BOTH) 
                {
                    BOOST_FOREACH(int j, nn_new[i]) 
                    {
                        BOOST_FOREACH(int k, rnn_old[i]) 
                        {
                            cc += update(j, k);
                        }
                        BOOST_FOREACH(int k, rnn_new[i]) 
                        {
                            cc += update(j, k);
                        }
                    }
                    BOOST_FOREACH(int j, nn_old[i]) 
                    {
                        BOOST_FOREACH(int k, rnn_new[i]) 
                        {
                            cc += update(j, k);
                        }
                    }
                }

#if NNDES_SHOW_PROGRESS
#ifdef _OPENMP
#pragma omp critical 
#endif
                ++progress;
#endif
            }

            cost += cc;

            /* It seems that using t as the number of new updates is
             * not accurate. But it doesn't matter much. 
             * Potential bug 2. ~Jichao */
            int t = 0;
#ifdef _OPENMP
#pragma omp parallel for default(shared) reduction(+:t)
#endif
            for (int i = 0; i < N; ++i) 
            {

                nn_old[i].clear();
                nn_new[i].clear();
                rnn_old[i].clear();
                rnn_new[i].clear();

                nn_full[i].clear();
                rnn_full[i].clear();

                // find the new ones
                for (int j = 0; j < K; ++j) 
                {
                    KNN::Element &e = nn[i][j];
                    if (e.key == KNN::Element::BAD)
                    {
                        assert(0); /* I think shouldn't be here */
                        continue;
                    }
                    if (e.flag)
                    {
                        nn_new[i].push_back(j);
                    }
                    else 
                    {
                        nn_old[i].push_back(e.key);
                    }

                    nn_full[i].push_back(e.key);
                }
                assert((int)nn_full[i].size() == K);

                t += nn_new[i].size();
                // // sample
                // if (nn_new[i].size() > unsigned(S)) 
                // {
                //     random_shuffle(nn_new[i].begin(), nn_new[i].end());
                //     nn_new[i].resize(S);
                // }
                if (nn_new[i].size() > unsigned(K)) 
                {
                    random_shuffle(nn_new[i].begin(), nn_new[i].end());
                    nn_new[i].resize(K);
                }
                BOOST_FOREACH(int &v, nn_new[i]) 
                {
                    nn[i][v].flag = false;
                    v = nn[i][v].key;
                }

                /* nn_old not sampled? Potential bug 3. ~Jichao */
                // // sample
                // if (nn_old[i].size() > unsigned(S)) 
                // {
                //     random_shuffle(nn_old[i].begin(), nn_old[i].end());
                //     nn_old[i].resize(S);
                // }
                if (nn_old[i].size() > unsigned(K)) 
                {
                    random_shuffle(nn_old[i].begin(), nn_old[i].end());
                    nn_old[i].resize(K);
                }
            }

            // symmetrize
            if ((option & GRAPH_RNN) || (option & GRAPH_BOTH)) 
            {
                for (int i = 0; i < N; ++i) 
                {
                    BOOST_FOREACH(int e, nn_old[i]) 
                    {
                        rnn_old[e].push_back(i);
                    }
                    BOOST_FOREACH(int e, nn_new[i]) 
                    {
                        rnn_new[e].push_back(i);
                    }

                    BOOST_FOREACH(int e, nn_full[i])
                    {
                        rnn_full[e].push_back(i);
                    }
                }
            }

#ifdef _OPENMP
#pragma omp parallel for default(shared) reduction(+:t)
#endif
            for (int i = 0; i < N; ++i) 
            {
                // if (rnn_old[i].size() > unsigned(S)) 
                // {
                //     random_shuffle(rnn_old[i].begin(), rnn_old[i].end());
                //     rnn_old[i].resize(S);
                // }
                // if (rnn_new[i].size() > unsigned(S)) 
                // {
                //     random_shuffle(rnn_new[i].begin(), rnn_new[i].end());
                //     rnn_new[i].resize(S);
                // }
                if (rnn_old[i].size() > unsigned(K)) 
                {
                    random_shuffle(rnn_old[i].begin(), rnn_old[i].end());
                    rnn_old[i].resize(K);
                }
                if (rnn_new[i].size() > unsigned(K)) 
                {
                    random_shuffle(rnn_new[i].begin(), rnn_new[i].end());
                    rnn_new[i].resize(K);
                }
            }

            return t;
        }

        void preNnf()
        {
            for (int n = 0; n < N; ++n)
            {
                nn_new[n].clear();
                nn_old[n].clear();
                rnn_new[n].clear();
                rnn_old[n].clear();

                /* it does not matter, if nn is not valid --- pre == 0 */
                for (int k = 0; k < K; ++k)
                {
                    nn[n][k].flag = true;
                }
            }
        }

        /* GraphOption must be GraphBoth */
        
        /* Edited by Arwa Wali by deleting compared becuse it is not used in the function */
        //void nnfAdjust(int pivot, DenMatSin& compared)
        void nnfAdjust(int pivot)
        {
            int p = pivot;
            // for (int n = 0; n < N; ++n)
            // {
            //     compared.unset(p, n);
            //     compared.unset(n, p);
            // }

            /* part 1 */
            /* for each KNN of p */
            for (int k = 0; k < K; ++k)
            {
                KNN::Element &e = nn[p][k];
                assert(e.key == nn_full[p][k]);
                /* update distance between p and e.key */
                e.dist = oracle(p, e.key);
            }

            /* re-sort p's KNN */
            sort(nn[p].begin(), nn[p].end(), compKnnEntries);
            /* for each p's RNN */
            vector<float> rnn_new_dists;
            for (int j = 0; j < (int) rnn_full[p].size(); ++j)
            {
                int key = rnn_full[p][j];
                KNN::iterator it;
                for (it = nn[key].begin(); it != nn[key].end(); ++it)
                {
                    if ( (*it).key == p )
                    {
                        break;
                    }
                }
                // assert(it != nn[key].end());
                if (it == nn[key].end()) continue;
                /* update distance and flag */
                (*it).dist = oracle(p, key);
                rnn_new_dists.push_back( (*it).dist );
                /* re-sort p's RNN's KNN */
                sort(nn[key].begin(), nn[key].end(), compKnnEntries);
            }

            /* part 2 */
            for (int k = 0; k < K; k++)
            {
                KNN::Element &e = nn[p][k];
                nn[e.key].update(KNN::Element(p, e.dist, true));
                // compared.set(p, e.key);
                // compared.set(e.key, p);
            }
            // for (int j = 0; j < (int) rnn_full[p].size(); ++j)
            // {
            //     int key = rnn_full[p][j];
            //     nn[key].update(KNN::Element(p, rnn_new_dists[j], true));
            //     compared.set(p, key);
            //     compared.set(key, p);
            // }
        }


        /* GraphOption must be GraphBoth */
        // void nnf(int pivot, DenMatSin & compared)
        void nnf(int pivot)
        {
            // /* update nn_full and rnn_full */
            // for (int n = 0; n < N; ++n)
            // {
            //     nn_full[n].clear();
            //     rnn_full[n].clear();
            //     for (int k = 0; k < K; ++k)
            //     {
            //         KNN::Element &e = nn[n][k];
            //         nn_full[n].push_back(e.key);
            //     }

            // }
            // for (int n = 0; n < N; ++n)
            // {
            //     for (int j = 0; j < K; ++j)
            //     {
            //         rnn_full[nn_full[n][j]].push_back(n);
            //     }
            // }
            for (int n = 0; n < N; ++n)
            {
                if (nn_full[n].size() > unsigned(K))
                {
                    // random_shuffle(nn_full[n].begin(), nn_full[n].end());
                    nn_full[n].resize(K);
                }
                if (rnn_full[n].size() > unsigned(K))
                {
                    //random_shuffle(rnn_full[n].begin(), rnn_full[n].end());
                    rnn_full[n].resize(K);
                }
            }




            
            int i = pivot;

            BOOST_FOREACH(int j, nn_full[i]) 
            {
                BOOST_FOREACH(int k, nn_full[i]) 
                {
                    if (j >= k) continue;
                    // if (compared.isSet(j,k)) continue;
                    update(j, k);
                    // compared.set(j,k);
                    // compared.set(k,j);
                }
                // BOOST_FOREACH(int k, rnn_full[i]) 
                // {
                //     // if (compared.isSet(j,k)) continue;
                //     update(j, k);
                //     // compared.set(j,k);
                //     // compared.set(k,j);
                // }
            }
            // BOOST_FOREACH(int j, rnn_full[i]) 
            // {
            //     BOOST_FOREACH(int k, rnn_full[i]) 
            //     {
            //         if (j >= k) continue;
            //         // if (compared.isSet(j,k)) continue;
            //         update(j, k);
            //         // compared.set(j,k);
            //         // compared.set(k,j);
            //     }
            // }

            /* update nn_full and rnn_full */
            for (int n = 0; n < N; ++n)
            {
                nn_full[n].clear();
                // rnn_full[n].clear();
                for (int k = 0; k < K; ++k)
                {
                    KNN::Element &e = nn[n][k];
                    nn_full[n].push_back(e.key);
                }

            }
            // for (int n = 0; n < N; ++n)
            // {
            //     for (int j = 0; j < K; ++j)
            //     {
            //         rnn_full[nn_full[n][j]].push_back(n);
            //     }
            // }


        }

        void postNnf()
        {
            for (int n = 0; n < N; ++n)
            {
                for (int k = 0; k < K; ++k)
                {
                    int key = nn[n][k].key;
                    nn_new[n].push_back(key);
                    rnn_new[key].push_back(n);
                }
                // if ((int) nn_new[n].size() > S)
                //     nn_new[n].resize(S);
                if ((int) nn_new[n].size() > K)
                    nn_new[n].resize(K);
            }
            for (int n = 0; n < N; ++n)
            {
                // if ((int) rnn_new[n].size() > S)
                //     rnn_new[n].resize(S);
                if ((int) rnn_new[n].size() > K)
                    rnn_new[n].resize(K);
            }
        }
    };

}

#endif 

