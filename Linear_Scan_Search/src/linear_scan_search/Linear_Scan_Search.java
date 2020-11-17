/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package linear_scan_search;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Vector;
import static linear_scan_search.Global1.Read_data_file;
import static linear_scan_search.Global1.Read_flags_file;
import static linear_scan_search.Global1.Read_label_file;
import static linear_scan_search.Global1.Read_sparse_data_file;

/**
 *
 * @author arwawali
 */
public class Linear_Scan_Search {

    /**
     * @param args the command line arguments
     */
    
    Vector<BitSet> data_flags;
    int Dim;
    int number_of_dist_computation;
    Linear_Scan_Search(Vector<BitSet> temp_flags, int temp_dim){
        this.data_flags=temp_flags;
        this.Dim= temp_dim;
    }
     // this unction compute the hammin distance only for bits that are set of query bit=1
    // this not reguler hamming distance which consider both differences in both bitsets 
    // with xor operation
     public float hamming_distance(BitSet query, BitSet fingerprint){
        BitSet xoroper=(BitSet) query.clone();
        fingerprint.flip(0, Dim);
        xoroper.and(fingerprint);
        
        return xoroper.cardinality();
    }
    public float simple_matching_coefficient_distance(BitSet query, BitSet fingerprint){
        BitSet andoper=(BitSet) query.clone();
        //System.out.println(query);
        //System.out.println(andoper);
        //System.out.println(fingerprint);
        // compute a
        andoper.and(fingerprint);
        //System.out.println(andoper);
        // compute d
        query.flip(0, Dim);
        //System.out.println(query);
        fingerprint.flip(0,Dim);
        query.and(fingerprint);
        //System.out.println(fingerprint);
        //System.out.println((((float) andoper.cardinality()+(float) query.cardinality())/(float) Dim));
        return (float) Math.sqrt(1-(((float) andoper.cardinality()+(float) query.cardinality())/(float) Dim));
        
    }
    ArrayList<result_list> Linear_Search(BitSet query, int K ){
        number_of_dist_computation=0;
        ArrayList<result_list> results=new ArrayList(K);
        int k=0;
        float dist;
        float d_max=Float.POSITIVE_INFINITY;
        //System.out.println(data_flags.size());
        for(int i=0; i<data_flags.size(); i++){
            
            //dist=  simple_matching_coefficient_distance(query,this.data_flags.get(i));
            dist=hamming_distance(query,this.data_flags.get(i));
            // System.out.println(dist);
            number_of_dist_computation++;
            if(dist <=d_max){
               
                result_list temp=new result_list(i,  dist,this.data_flags.get(i) );
                    if(k<K){

                        results.add(temp);
                        if(k==K-1){
                            Collections.sort(results);
                            d_max=results.get(K-1).dis;
                        }
                        k++;

                    }
                    else{
                                    //if(dist<results.get(K-1).dis){
                         //System.out.println("I *** "+ temp.objects_index);
                         results.set(K-1, temp);
                         //results.get(K-1).dis=temp.dis;//.set(K-1, temp);
                         //results.get(K-1).objects_index=temp.objects_index;
                         //results.get(K-1).finger_prints=temp.finger_prints;
                         Collections.sort(results);
                         d_max=results.get(K-1).dis;
                                    //result_list temp=new result_list(list_toscan[i], dist,fingerprints.get(i) );
                                    //}


                    }
            }
        } 
        return results ; 
    }
    public int get_number_of_dist_computation(){
        return this.number_of_dist_computation;
    }
    public static void main(String[] args) throws FileNotFoundException {
        // TODO code application logic here
      String data_file_path=args[0];
      
      //String data_file_path="/Users/arwawali/Google Drive/ResearchDataSets/movement/";
    
      // the data set has to have to first rows that has dimentions (rows cols)
      String dataset_name=args[1];//"faces";
      //String dataset_name="movement";
      
      
      int K_dash=Integer.parseInt(args[2]);
      int K_less=10;
      //int K_dash=200;
      float data[][]=Read_data_file(data_file_path+ dataset_name + ".dvf");
      float sparse_data[][]=Read_sparse_data_file(data_file_path+ dataset_name + ".sparse", data.length, data[0].length);
       int rows=data.length;
      int cols=data[0].length;
      // Read label files only for measuring accuracy
      int labels[]=Read_label_file(data_file_path+ dataset_name + ".matlab.lab", data.length);
      
      
      
       Vector<BitSet> data_flags=Read_flags_file(data_file_path+ dataset_name + ".flags", rows, cols);
     
       
      
      int[] idx=new int[data.length];
      
      for(int i =0; i<data.length; i++){
          idx[i]=i;
      }
       BitSet query;
       long startTime, endTime, totalTime;
      ArrayList<knn_dis_pair> sparse_results, actual_results;
      float dis, K_dis;
      ArrayList<Integer> inds;
      float label_acuracy,sparse_accuracy, actual_accuracy, distance_accuracy;
      Linear_Scan_Search LSS=new Linear_Scan_Search(data_flags, cols);
      float avg_label_acuracy=0.0f,avg_sparse_accuracy=0.0f, avg_actual_accuracy=0.0f, avg_distance_accuracy=0.0f, avg_totalTime=0.0f, avg_distances_computations=0.0f;
      float total_label_acuracy,total_sparse_accuracy, total_actual_accuracy,total_actual_accuracy_less=0.0f, total_distance_accuracy, total_totalTime, total_distances_computations;
      //for(int iter=0; iter<100; iter++){
          // pick 1000 quiries from sparsified data
      
          Collections.shuffle(Arrays.asList(idx));
          total_label_acuracy=0.0f;
          total_sparse_accuracy=0.0f;
          total_actual_accuracy=0.0f;
          total_distance_accuracy=0.0f;
          total_totalTime=0.0f;
          total_distances_computations=0.0f;
      for(int i=0; i<Math.min(1000,idx.length); i++){
           System.out.println("Trial # " + Integer.toString(i+1));
           // get the query BitSet 
          query=data_flags.get(idx[i]);
          startTime = System.currentTimeMillis();
          ArrayList<result_list> results=LSS.Linear_Search(query, K_dash );
          endTime   = System.currentTimeMillis();
          totalTime = endTime - startTime;
          
          //1.  measure the Accuracy in term of labels
          label_acuracy=0.0f;
          for(int j=0; j<results.size(); j++){
              //System.out.println(results.get(j).objects_index);
              if(labels[idx[i]]== labels[results.get(j).objects_index]){
                  label_acuracy++;
              }    
          }
          // devide the label accuracy by K_dash
          label_acuracy=label_acuracy/(float) K_dash;
          System.out.println("Label Accuracy: " + Float.toString(label_acuracy));
          //********** System.out.println("Label Accuracy: " + label_acuracy);
              total_label_acuracy=total_label_acuracy+label_acuracy;
              //2.  measure the time
           System.out.println("Time: " + Long.toString(totalTime) );
              total_totalTime=total_totalTime+totalTime;
          //3.  measure the number of distances calculations 
          System.out.println("Number of distances " + Integer.toString(LSS.get_number_of_dist_computation())  );
          total_distances_computations=total_distances_computations+ LSS.get_number_of_dist_computation();
              
          //4.  find the K_dash points using the sparsified features for the query
          sparse_results= new ArrayList();
          for(int j=0; j<rows; j++){
              dis=distance(sparse_data[idx[i]],sparse_data[j] );
              knn_dis_pair temp=new knn_dis_pair(j,dis );
              sparse_results.add(temp);
          }
          Collections.sort(sparse_results);
          inds=new ArrayList();
          for(int j=0; j<K_dash; j++){
              inds.add(sparse_results.get(j).getIndex());
          }
          // find the number of matches between the sparsfied points results and the fingerprint query results
          sparse_accuracy=0.0f;
          for(int j=0; j<K_dash; j++){
              //System.out.println(results.size());
              if(inds.contains((Integer) results.get(j).objects_index)){
                  sparse_accuracy++;
              }
          }
          
         System.out.println("Number of points that match the points with sparse distances in the results= " + Float.toString(sparse_accuracy/(float) K_dash)  );
         total_sparse_accuracy=total_sparse_accuracy+(sparse_accuracy/(float) K_dash);
              
          
          //5.  find the K_dash points using the actual features for the query
          actual_results= new ArrayList();
          for(int j=0; j<rows; j++){
              dis=distance(data[idx[i]], data[j]);
              knn_dis_pair temp=new knn_dis_pair(j,dis );
              actual_results.add(temp);
          }
          Collections.sort(actual_results);
          inds=new ArrayList();
          for(int j=0; j<K_dash; j++){
              inds.add(actual_results.get(j).getIndex());
          }
          // find the number of matches between the sparsfied points results and the fingerprint query results
          actual_accuracy=0.0f;
          for(int j=0; j<K_dash; j++){
              if(inds.contains((Integer) results.get(j).objects_index)){
                  actual_accuracy++;
              }
          }
          
           System.out.println("Number of points that match the points with actual distances in the results= " + Float.toString(actual_accuracy/(float) K_dash ) );
           total_actual_accuracy=total_actual_accuracy+(actual_accuracy/(float) K_dash);
            
           
              inds=new ArrayList();
              for(int j=0; j<K_less; j++){
                  inds.add(actual_results.get(j).getIndex());
              }
              // find the number of matches between the actual points results and the fingerprint query results
              actual_accuracy=0.0f;
              for(int j=0; j<K_less; j++){
                  if(inds.contains((Integer) results.get(j).objects_index)){
                      actual_accuracy++;
                  }
              }
              
              System.out.println("Number of points that match the points with actual distances in the results devided by k_less = " + Float.toString(actual_accuracy/(float) K_less)  );
              total_actual_accuracy_less=total_actual_accuracy_less+(actual_accuracy/(float) K_less);
             
          // 6. Number of points  with distance less than actual K_dashth distance 
          K_dis=actual_results.get(K_dash-1).getDis();
          distance_accuracy=0.0f;
          for(int j=0; j<results.size(); j++){
              if(distance(data[idx[i]], data[results.get(j).objects_index])<=K_dis)
                  distance_accuracy++;
          }
          
          System.out.println("Number of points with actual distances in the results less than K-th actual distances = " + Float.toString(distance_accuracy/(float) K_dash  ));
          total_distance_accuracy=total_distance_accuracy+(distance_accuracy/(float) K_dash );

        //}
          //float avg_label_acuracy=0.0f,avg_sparse_accuracy=0.0f, avg_actual_accuracy=0.0f, avg_distance_accuracy=0.0f, avg_totalTime=0.0f, avg_distances_computations=0.0f;
    
          //avg_label_acuracy+=(total_label_acuracy/(float)Math.min(1000,idx.length) );
          //avg_sparse_accuracy+=(total_sparse_accuracy/(float)Math.min(1000,idx.length) );
          //avg_actual_accuracy+=(total_actual_accuracy/(float)Math.min(1000,idx.length) );
          //avg_distance_accuracy+=(total_distance_accuracy/(float)Math.min(1000,idx.length) );
          //avg_totalTime+=(total_totalTime/(float)Math.min(1000,idx.length) );
          //avg_distances_computations+=(total_distances_computations/(float)Math.min(1000,idx.length) );  
          /*
          System.out.println("Trial # " + i+1);
          System.out.println("Label Accuracy: " + (total_label_acuracy/(float)Math.min(1000,idx.length)));
          System.out.println("Time: " + total_totalTime/(float)Math.min(1000,idx.length) );
          System.out.println("Number of distances " + total_distances_computations/(float)Math.min(1000,idx.length)  );
          System.out.println("Number of points that match the points with sparse distances in the results= " + total_sparse_accuracy/(float)Math.min(1000,idx.length)  );
          System.out.println("Number of points that match the points with actual distances in the results= " + total_actual_accuracy/(float)Math.min(1000,idx.length) );
          System.out.println("Number of points with actual distances in the results less than K-th actual distances = " + total_distance_accuracy/(float)Math.min(1000,idx.length)  );
          */
      }
      System.out.println("Average Results:");
      System.out.println("Average Label Accuracy: " + total_label_acuracy/(float) 1000);
      System.out.println("Average Time: " + total_totalTime/(float) 1000 );
      System.out.println("Average Number of distances " + total_distances_computations/ (float) 1000  );
      System.out.println("Average Number of points that match the points with sparse distances in the results= " + total_sparse_accuracy/(float) 1000  );
      System.out.println("Average Number of points that match the points with actual distances in the results= " + total_actual_accuracy/(float) 1000 );
      System.out.println("Average Number of points that match the points with actual distances devided by K_less=100 in the results= " + total_actual_accuracy_less/(float) 1000 );
      System.out.println("Average Number of points with actual distances in the results less than K-th actual distances = " + total_distance_accuracy/(float) 1000  );
        
    }
    
    /* this function calculate the distance between each two vectors and return the results */
    public static float distance(float[] v1, float[] v2) {
         float res=0.0f;
         
             for(int i=0; i<v1.length; i++){
         
                     res+=Math.pow((v1[i] - v2[i]),2);
                 
            }
         return (float) Math.sqrt(res);
    }
    public static float distances_getValue_bitVector(float[] v1, float[] v2, BitSet bitset){
        
        float res=0.0f;
         
             for(int i=0; i<v1.length; i++){
                    if(bitset.get(i)==true){
                         res+=Math.pow((v1[i]-v2[i]), 2);
                    }
                 
            }
         return (float) Math.sqrt(res);
        
        
    }
}
