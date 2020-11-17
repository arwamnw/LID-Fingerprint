/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package linear_scan_search;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Scanner;
import java.util.Vector;
import java.util.concurrent.ExecutionException;

/**
 *
 * @author arwawali
 */
public class Global1 {
    int[] labels;
    static float[][] data;
    float[] distance_matrix;
    static Vector<Vector<Float>> distance_matrix1;
    //float[][] sparse_data;
    // assignment matrix is not symmetrical becuse we use L-random walk to find it values. (1-l_{ij}) (xi-xj)^2
    float[][] assignment_matrix;
    Vector<Vector<Float>> similarity_matrix;
    float[] degree_matrix;
    float[][] laplcian_matrix;
    Vector<Vector<Float>> laplcian_matrix1;
     int rows;
    static int cols;
    // variables means and standard deviations to store the means and std for each coulumn in the dataset
    float[] means;
    float[] standard_deviation;
    Vector<Vector<Integer>> KNN;
    
    // constructor
    public Global1(float[][] data_temp, Vector<Vector<Float>> temp_distances, int[] labels_temp){
        this.data=data_temp;
        this.labels=labels_temp;
        this.rows=this.data.length;
        //int temprows=(( this.rows*( this.rows+1))/ 2)- this.rows;
        //System.out.println(temprows);
        this.cols=this.data[0].length;
        //this.distance_matrix=new float[ temprows];
        this.similarity_matrix=new Vector<Vector<Float>>();
        this.laplcian_matrix1=new Vector<Vector<Float>>();
        if(temp_distances != null){
                    this.distance_matrix1=temp_distances;
                    //System.out.println("****" + this.distance_matrix1.size());
        }
        this.degree_matrix=new float[this.rows];
        //this.laplcian_matrix=new float[this.rows][this.rows];
        this.means=new float[this.cols];
        this.standard_deviation=new float[this.cols];
     
    }
    
    /* this function comput the distance between data point and any external point
    */
    
    public float distance_between_external_point_and_data_point(float[] external, int point_index) {
         float res=0.0f;
         
             for(int i=0; i<external.length; i++){
         
                     res+=Math.pow((external[i] - this.data[point_index][i]),2);
                 
            }
         return (float) Math.sqrt(res);
    }
    public float distance_between_external_point_and_data_point_Subspace(float[] external, int point_index, BitSet bitset) {
         float res=0.0f;
         
             for(int i=0; i<external.length; i++){
                    if(bitset.get(i)==true){
                        res+=Math.pow((external[i] - this.data[point_index][i]),2);
                    }
                 
            }
         return (float) Math.sqrt(res);
    }
    /*  This function compute the avarage point of the cluster */
   public float[] compute_the_avarage_point_of_group(ArrayList<Integer> group){
       // initilize to 0
       float[] average_point= new float[cols];
       for(int i=0; i< cols; i++){
            average_point[i]=0.0f;
        }
        for (Integer point : group) {
            for(int i=0; i< cols; i++){
                average_point[i]+=this.data[point][i];
            }
        }
        
        for(int i=0; i< cols; i++){
            average_point[i]=average_point[i]/group.size();
        }
        
        return average_point;
   }
  /*  This function compute the avarage point of the cluster */
   public float[] compute_the_avarage_point_of_group_Subspace(ArrayList<Integer> group, BitSet bitset){
       // initilize to 0
       float[] average_point= new float[cols];
       for(int i=0; i< cols; i++){
            average_point[i]=0.0f;
        }
        for (Integer point : group) {
            for(int i=0; i< cols; i++){
                 if(bitset.get(i)==true){
                      average_point[i]+=this.data[point][i];
                 }
            }
        }
        
        for(int i=0; i< cols; i++){
            if(bitset.get(i)==true){
                average_point[i]=average_point[i]/group.size();
            }
        }
        
        return average_point;
   }
    /* this function calculate the distance between each two vectors and return the results */
    public float distance(float[] v1, float[] v2) {
         float res=0.0f;
         
             for(int i=0; i<v1.length; i++){
         
                     res+=Math.pow((v1[i] - v2[i]),2);
                 
            }
         return (float) Math.sqrt(res);
    }
    
    /* this function calculate the similarity distance between each two vectors and return the results 
    by devide each column diffrence by 2*standard deviation of that column*/
    public float similarity_distance(float[] v1, float[] v2) {
         float res=0.0f;
         
             for(int i=0; i<v1.length; i++){
                    if(standard_deviation[i]!=0)
                        res+=((-1)*Math.pow((v1[i] - v2[i]),2)/(2*standard_deviation[i]));
                    else{
                        res+=((-1)*Math.pow((v1[i] - v2[i]),2)/(2));
                    }
                 
             }
         return res;
    }
    
    /* this function calculate the distance between each two 
    vectors and store the result in one dimentional array (define for symmetric matrix without the diagonal)    
    */
    public void distance_matrix(){
        int k = 0;
        for(int i=1; i<this.rows; i++){
            
            for (int j=0; j<i; j++){
                
                
                  this.distance_matrix[k]=distance(this.data[i],this.data[j]);
                  k++;
                
                
            }
        }
        
            
    }
      /* this function calculate the distance between each two vectors and return the results */
    public float distance_index(int obj1_indx , int obj2_indx) {
         float res=0.0f;
         
             for(int i=0; i<this.data[obj1_indx].length; i++){
                    //System.out.println(this.data[obj1_indx][i]);
                     res+=(float) Math.pow((this.data[obj1_indx][i] - this.data[obj2_indx][i]),2);
                 
            }
             //System.out.println(res);
         return (float) Math.sqrt(res);
    }
    public float distances_getValue_bitVector(int med, int obj, BitSet bitset){
        
        float res=0.0f;
         
             for(int i=0; i<this.data[med].length; i++){
                    if(bitset.get(i)==true){
                         res+=Math.pow((this.data[med][i]-this.data[obj][i]), 2);
                    }
                 
            }
         return (float) Math.sqrt(res);
        
        
    }
    
    public void distance_matrix_faster(){
        Vector<Float> temp=null;
       for(int i=0; i<this.rows; i++){
            temp=new Vector<Float>();
            for (int j=0; j<=i; j++){
                
                //if(i==j)
                //   this.similarity_matrix.get(i).add(1.0d);
                //else 
                
                temp.add(distance(this.data[i],this.data[j]));
                
                
                
            }
            //System.out.println(temp.size());
            this.distance_matrix1.add(temp);
        }
        
            
    }
    
    public float distances_getValue(int i, int j){
        
        if(j<i)
            return this.distance_matrix[j*(rows-1) - (j-1)*((j-1) + 1)/2 + i - j - 1]; 
        else if(i<j)
           return this.distance_matrix[i*(rows-1) - (i-1)*((i-1) + 1)/2 + j - i - 1]; 

        return 0.0f;
        
    }
    
    public float distances_getValue1(int i, int j){
        
        if(j<=i){
                //System.out.println(i + " " + j);
                return this.distance_matrix1.get(i).get(j);
        }
        else if(i<j)
           return this.distance_matrix1.get(j).get(i);

        return 0.0f;
        
    }
    
    public float laplasian_get_absoulte_Value(int i, int j){
        
        if(j<=i)
            return Math.abs(this.laplcian_matrix1.get(i).get(j)); 
        else if(i<j)
           return Math.abs(this.laplcian_matrix1.get(j).get(i));

        return 0.0f;
        
    }
    // function to calculate the means and standard deviation of data columns
    public void columns_means_standarddeviation(){
        
        // calculate the means for all columns
       
        for(int j=0; j<this.cols; j++){
           for(int i=0; i<this.rows; i++){
               if(i==0)
                 means[j]=(this.data[i][j]/this.rows);
               else
                 means[j]+=(this.data[i][j]/this.rows);  
           }
        }
        //calacula standard deviation
        for(int j=0; j<this.cols; j++){
           for(int i=0; i<this.rows; i++){
               if(i==0)
                 standard_deviation[j]=((float) Math.pow(this.data[i][j]-means[j],2))/(this.rows);
               else
                 standard_deviation[j]+=((Math.pow(this.data[i][j]-means[j],2))/(this.rows));  
           }
        }
        
    }
    
    /* this function calculate find TRINGULE of similiarty matrix between each two vectors in the data set, and store the result in tringuler matrix*/
    public void gussian_similaity_matrix(){
        
       Vector<Float> temp=null;
       for(int i=0; i<this.rows; i++){
            temp=new Vector<Float>();
            for (int j=0; j<=i; j++){
                
                //if(i==j)
                //   this.similarity_matrix.get(i).add(1.0d);
                //else 
                
                temp.add((float) Math.pow(Math.E,  similarity_distance(this.data[i],this.data[j])));
                
                
                
            }
            this.similarity_matrix.add(temp);
        }
       
        
    }
    
    public void degree_matrix(){
        
        for(int i=0; i<this.rows; i++){
            this.degree_matrix[i]=0;
            for(int j=0; j<this.rows; j++){
                if(j<i)
                   this.degree_matrix[i]+= this.similarity_matrix.get(i).get(j);
                else if(j>i)
                   this.degree_matrix[i]+= this.similarity_matrix.get(j).get(i); 
                   
                    
            }
        
        }
        
    }
    /* laplcian_matrix is symmetric , 
    
    when we find the laplacian between mediod vk and object xi, we always check the lki, not lik, 
    becuase we want to see if xi is among nearest neighbor of vk  
    */
    public void laplcian_matrix(){
        
        for(int i=0; i<this.rows; i++){
            
            for(int j=0; j<this.rows; j++){
                
                if(i==j && this.degree_matrix[i]!=0.)
                    this.laplcian_matrix[i][j]=1;
               
                else if(j<i && this.degree_matrix[i]!=0. && this.degree_matrix[j]!=0. ){
                    
                    this.laplcian_matrix[i][j]=((-1)* this.similarity_matrix.get(i).get(j))/(float) Math.sqrt(this.degree_matrix[i]*this.degree_matrix[j]);
                }
                else if(j>i && this.degree_matrix[i]!=0 && this.degree_matrix[j]!=0. ){
                    this.laplcian_matrix[i][j]=((-1)* this.similarity_matrix.get(j).get(i))/(float) Math.sqrt(this.degree_matrix[i]*this.degree_matrix[j]);
                }
                else{
                    this.laplcian_matrix[i][j]=0;
                }
 
            }
        }    
        
        
    }
    
        public void laplcian_matrix_faster(){
        Vector<Float> temp=null;
        for(int i=0; i<this.rows; i++){
            temp=new Vector<Float>();
            for(int j=0; j<=i; j++){
                
                if(i==j && this.degree_matrix[j]!=0.)
                    temp.add(1.0f);
                    //this.laplcian_matrix[i][j]=1;
                else if(i!=j && this.degree_matrix[i]!=0. && this.degree_matrix[j]!=0.){
                    //System.out.println(Math.sqrt(this.degree_matrix[i]*this.degree_matrix[j]));
                    if(((-1)* this.similarity_matrix.get(i).get(j))/Math.sqrt(this.degree_matrix[i]*this.degree_matrix[j])!= Float.NEGATIVE_INFINITY){
                        temp.add(((-1)* this.similarity_matrix.get(i).get(j))/(float) Math.sqrt(this.degree_matrix[i]*this.degree_matrix[j]));
                    }
                    else{
                        temp.add(0.f);
                    }   
                }
                      
                else{
                    temp.add(0.f);
                }
 
            }
            this.laplcian_matrix1.add(temp);
        }    
        
        
        
        
    }
        
    // this function is update the distance matrix to include the laplacian values (1-L)(x-y)
    public void  update_distance_matrix_with_laplacian(){
        float temp;
        for(int i=0; i<this.rows; i++){
            //temp=new Vector<Float>();
            for (int j=0; j<=i; j++){
                temp=this.distance_matrix1.get(i).get(j)*(1-laplasian_get_absoulte_Value(i,j));
                this.distance_matrix1.get(i).set(j, temp);
                
                
                //this.distance_matrix1.s
                
            }
            
        }
    }
    
    public void normalization_function(){
       
       
       float[] means=new float[this.cols];
       float[] standard_deviation=new float[this.cols];
       
       // calculate the means for all columns
       
        for(int j=0; j<this.cols; j++){
           for(int i=0; i<this.rows; i++){
               if(i==0)
                 means[j]=(this.data[i][j]/this.rows);
               else
                 means[j]+=(this.data[i][j]/this.rows);  
           }
        }
        //calacula standard deviation
        for(int j=0; j<this.cols; j++){
           for(int i=0; i<this.rows; i++){
               if(i==0)
                 standard_deviation[j]=((float) Math.pow(this.data[i][j]-means[j],2))/(this.rows);
               else
                 standard_deviation[j]+=((Math.pow(this.data[i][j]-means[j],2))/(this.rows));  
           }
        }
        
        // normalize the data
        for(int i=0; i<this.rows; i++){
           for(int j=0; j<this.cols; j++){
               
                if(!Float.isNaN((this.data[i][j]-means[j])/((float) Math.sqrt(standard_deviation[j]))))
                  this.data[i][j]=(this.data[i][j]-means[j])/((float) Math.sqrt(standard_deviation[j]));
               else
                   this.data[i][j]=0.f;
           }
        }
       
       
       
       
   }
    
    public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException, ExecutionException{
       
      //String data_file_path=args[0];
   
      
      String data_file_path="/Users/arwawali/Documents/NetBeansProjects/Subspace_K_mediods/src/subspace_k_mediods/fake_data_norm.dvf";
      
      
      //String KNN_file="/Users/arwawali/Documents/NetBeansProjects/Subspace_K_mediods/src/subspace_k_mediods/orl_norm.knn.1";
      
      // the data set has to have to first rows that has dimentions (rows cols)
      //String dataset_name=args[1];//"faces";
     
      float data[][]=Read_data_file(data_file_path);
       // Read label files only for measuring accuracy
      //int labels[]=Read_label_file(data_file_path+ dataset_name + ".matlab.lab", data.length);
     
      /* for(int i=0; i<data.length; i++){
           for(int j=0; j<data[0].length; j++){
               
               System.out.print(data[i][j]+" ");
           }
           System.out.print("\n");
        } */
       
       System.out.print("************************************* \n");
      
       
       /*Vector<Vector<Integer>> tempKNN=Read_KNN_Matrix(KNN_file, data.length);
       for(int i=0; i<tempKNN.size(); i++){
           for(int j=0; j<tempKNN.get(0).size(); j++){
               
               System.out.print(tempKNN.get(i).get(j)+" ");
           }
           System.out.print("\n");
        }*/
       Vector<Vector<Float>> distances=null;
      File distance_file=new File(data_file_path + ".dd.1");
      if(distance_file.exists()){
           distances=Read_Distance_matrix(data_file_path + ".dd.1", data.length);
      } 
      //  Global1 g=new Global1(data, distances);
      
      // recaluclate distance matrix using new data (sparse_data)
     /*   g.distance_matrix_faster();
         for(int i=0; i<g.rows; i++){
            for (int j=0; j<=i; j++){
                System.out.print(g.distances_getValue1(i, j) + " ");
            }
            System.out.println("\n");
        }
         
        System.out.println("\n");
        /* calcualte laplacian matrix by calling all required functions
         1.  calculate means + standard deviations */
        //g.columns_means_standarddeviation();
        //for(int i=0; i<g.standard_deviation.length; i++){  
        //    System.out.print(g.standard_deviation[i] + " ");
        //}
        //System.out.println("\n");
        // 2. calaculate gussian similarity
        //g.gussian_similaity_matrix();
       /*for(int i=0; i<g.rows; i++){
            for (int j=0; j<=i; j++){
                System.out.print(g.similarity_matrix.get(i).get(j) + " ");
            }
            System.out.println("\n");
        } */
        //System.out.println("\n");
        //3.  caluclate degree matrix
        //g.degree_matrix();
        //for(int i=0; i<g.degree_matrix.length; i++){  
        //    System.out.print(g.degree_matrix[i] + " ");
        //}
       // System.out.println("\n");
        // 4. calcualte laplcaian matrix
        //g.laplcian_matrix();
       // g.laplcian_matrix_faster();
        /*for(int i=0; i<g.rows; i++){
            
            for(int j=0; j<g.rows; j++){
                
               System.out.print(g.laplcian_matrix[i][j] + " "); 
            }
            System.out.println("\n");
        
        } */
        
        
        //for(int i=0; i<g.rows; i++){
        //    for (int j=0; j<=i; j++){
        //        System.out.print(g.laplasian_get_absoulte_Value(i, j) + " ");
        //    }
        //    System.out.println("\n");
       // } 
        
       // g.update_distance_matrix_with_laplacian();
        
       // for(int i=0; i<g.rows; i++){
        //    for (int j=0; j<=i; j++){
       //         System.out.print(g.distances_getValue1(i, j) + " ");
       //     }
       //     System.out.println("\n");
       // }  
        
        
       
   }
      // This function is to read ANN using NN-Descent
        
    public static int[][] Read_KNN_Matrix(String KNN_file, int rows) throws FileNotFoundException{
        
        int[][] matrix = {{1}, {2}};

        File inFile = new File(KNN_file);
        Scanner in = new Scanner(inFile);

        int intLength = 0;
        
          
             String[] length = in.nextLine().trim().split("\\s+");
             for (int i = 0; i < length.length; i++) {
               intLength++;
             }
        

        //in.close();

        matrix = new int[rows][Integer.parseInt(length[1])];
        in = new Scanner(inFile);
        System.out.println(rows);
        System.out.println(Integer.parseInt(length[1]));
        int lineCount = 0;
        while (in.hasNextLine() && lineCount<rows) {
          String[] currentLine = in.nextLine().trim().split("\\s+"); 
          
          //System.out.println(lineCount);
             for (int i = 0; i < currentLine.length; i++) {
                matrix[lineCount][i] = Integer.parseInt(currentLine[i]);    
                    }
          lineCount++;
         }                                 
         return matrix;
        
        
        
    }
    
     
    public static int[] Read_label_file(String labels_file, int labels_length) throws FileNotFoundException{
         int[] matrix;

         File inFile = new File(labels_file);
         Scanner in = new Scanner(inFile);
         


         //String[] rows_cols = in.nextLine().trim().split("\\s+");
    
         //int cols=Integer.parseInt(rows_cols[1]);
         //int rows=Integer.parseInt(rows_cols[0]);
    
    

         matrix = new int[labels_length];
         int lineCount = 0;
         while (in.hasNextLine() && lineCount<labels_length) {
           String[] currentLine = in.nextLine().trim().split("\\s+"); 
           matrix[lineCount] = Integer.parseInt(currentLine[0]);  
           
           lineCount++;
          }                                 
          return matrix;

    

    }
    
    public static List<ArrayList> Read_RNN_Matrix(String rnn_file, int rows) throws FileNotFoundException{
        
        List<ArrayList> RNN= new ArrayList<ArrayList>();
        File inFile = new File(rnn_file);
         Scanner in = new Scanner(inFile);
         int lineCount = 0;
        while (in.hasNextLine() && lineCount<rows) {
          String line=in.nextLine().trim();
          ArrayList inner_list=new ArrayList();
          if(!line.isEmpty()){
            String[] currentLine = line.split("\\s+"); 
             for (int i = 0; i < currentLine.length; i++) {
                 if(currentLine[i]==" "){
                     inner_list.add(-1);
                     continue;}
               inner_list.add(Integer.parseInt(currentLine[i]));    
             }
             RNN.add(inner_list);
             lineCount++;
           }
          else{
              inner_list.add(-1);
              RNN.add(inner_list);
          }
         }                                 
         
        return RNN;
        
    }
        // Function to read the data file
   public static float[][] Read_data_file(String data_file) throws FileNotFoundException{

         float[][] matrix = {{1}, {2}};

         File inFile = new File(data_file);
         Scanner in = new Scanner(inFile);
         


         String[] rows_cols = in.nextLine().trim().split("\\s+");
    
         int cols=Integer.parseInt(rows_cols[1]);
         int rows=Integer.parseInt(rows_cols[0]);
    
    

         matrix = new float[rows][cols];
         int lineCount = 0;
         while (in.hasNextLine() && lineCount<rows) {
           String[] currentLine = in.nextLine().trim().split("\\s+"); 

           //System.out.println(currentLine);
              for (int i = 0; i < currentLine.length; i++) {
                 matrix[lineCount][i] = Float.parseFloat(currentLine[i]);    
                     }
           lineCount++;
          }                                 
          return matrix;



     }
     public static Vector<Vector<Float>> Read_Distance_matrix(String data_file, int data_lenght) throws FileNotFoundException{
         Vector<Vector<Float>> matrix= new Vector<Vector<Float>>();
         Vector<Float> temp=null;
         File inFile = new File(data_file);
         
         Scanner in = new Scanner(inFile);
         int lineCount = 0;
         while (in.hasNextLine() && lineCount<data_lenght) {
             String[] currentLine = in.nextLine().trim().split("\t"); 
             //System.out.println(currentLine.length);
             
             temp=new Vector<Float>();
              for (int i = 0; i < currentLine.length; i++) {
                  //System.out.print(Float.parseFloat(currentLine[i]) + " ");
                  temp.add(Float.parseFloat(currentLine[i]));
              }
              //System.out.println(temp);
              matrix.add(temp);
              //System.out.println(temp.size());

              lineCount++;
         }
         return matrix;
         
     }
             // Function to read the data file
   public static float[][] Read_sparse_data_file(String data_file, int rows, int cols) throws FileNotFoundException{

         float[][] matrix = {{1}, {2}};

         File inFile = new File(data_file);
         Scanner in = new Scanner(inFile);
         

         matrix = new float[rows][cols];
         int lineCount = 0;
         while (in.hasNextLine() && lineCount<rows) {
           String[] currentLine = in.nextLine().trim().split("\\s+"); 

           //System.out.println(currentLine);
              for (int i = 0; i < currentLine.length; i++) {
                 matrix[lineCount][i] = Float.parseFloat(currentLine[i]);    
                     }
           lineCount++;
          }                                 
          return matrix;



     }    
      // Function the flags file
   public static Vector<BitSet> Read_flags_file(String sparse_file, int rows, int cols) throws FileNotFoundException{

         Vector<BitSet> matrix=new Vector<BitSet>();// = {{1}, {2}};

         File inFile = new File(sparse_file);
         Scanner in = new Scanner(inFile);
         
         //matrix = new float[rows][cols];
         int lineCount = 0;
         while (in.hasNextLine() && lineCount<rows) {
           String[] currentLine = in.nextLine().trim().split("\\s+"); 

           //System.out.println(currentLine);
              BitSet bitset=new BitSet(cols);
              for (int i = 0; i < currentLine.length; i++) {
                 if(Integer.parseInt(currentLine[i])==1){
                     bitset.set(i, true);
                 }
                 else{
                     bitset.set(i, false);
                 }
                 
                     }
           matrix.add(bitset);   
           lineCount++;
          }                                 
          return matrix;



     }
    
    
    
}
