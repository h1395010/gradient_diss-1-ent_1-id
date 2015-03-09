package return_of_gradient_descent;

import java.text.DecimalFormat;
import java.util.Set;

import com.google.common.collect.Table;
import com.google.common.collect.Table.Cell;


public class GradientDescent 
{
	  static int MAX_ITER = 100;
	  static double LEARNING_RATE = 0.1;           
	  
    static final String LABEL = "atheism";
    //static final String LABEL = "sports";
    //static final String LABEL = "science";
	  
	  public static void gradient_descent( Table< int[] , String , Integer > train_freq_count_against_globo_dict,
			  					           Table< int[] , String , Integer > test_freq_count_against_globo_dict,
			  						       Set<String> GLOBO_DICT )
	  {
		  int globo_dict_size = GLOBO_DICT.size();
		  int number_of_files__train = train_freq_count_against_globo_dict.size();
		  
		  double[] theta = new double[ globo_dict_size + 1 ];//one for bias
		  for (int i = 0; i < theta.length; i++) 
		  {
			  theta[i] = randomNumber(0,1);
		  }
		    
		  		    

		   double[][] feature_matrix__train = new double[ number_of_files__train ][ globo_dict_size ];
		   double[] outputs__train = new double [ number_of_files__train ];
		    
		   int z = 0;
		   for ( Cell< int[] , String , Integer > cell: train_freq_count_against_globo_dict.cellSet() )
		   {			   
			   int[] container_of_feature_vector = cell.getRowKey();
			   
			   for (int q = 0; q < globo_dict_size; q++) 
	           {
				   feature_matrix__train[z][q] = container_of_feature_vector[q];
	           }
			   outputs__train[z] = String.valueOf( cell.getColumnKey() ).equals(LABEL) ? 1.0 : 0.0;
	           
	           z++;
		   }
	
	  double cost, error, hypothesis;
	  double[] gradient;
	  int p, iteration;

	  iteration = 0;
	  do 
	  {
	    iteration++;
	    error = 0.0;
	    cost = 0.0;
	    
	    //loop through all instances (complete one epoch)
	    for (p = 0; p < number_of_files__train; p++) 
	    {
	    	
	      // 1. Calculate the hypothesis h = X * theta
	      hypothesis = calculateHypothesis( theta, feature_matrix__train, p, globo_dict_size );

	      // 2. Calculate the loss = h - y and maybe the squared cost (loss^2)/2m
	      cost = hypothesis - outputs__train[p];
	      
	      // 3. Calculate the gradient = X' * loss / m
	      gradient = calculateGradent( theta, feature_matrix__train, p, globo_dict_size, cost, number_of_files__train);
	      
	      // 4. Update the parameters theta = theta - alpha * gradient
	      for (int i = 0; i < globo_dict_size; i++) 
	      {
	    	  theta[i] = theta[i] - LEARNING_RATE * gradient[i];
	      }

	    }
	    
		//summation of squared error (error value for all instances)
	    error += (cost*cost);	    
	  
	  /* Root Mean Squared Error */
	  if (iteration < 10) 
		  System.out.println("Iteration 0" + iteration + " : RMSE = " + Math.sqrt(  error/number_of_files__train  ) );
	  else
		  System.out.println("Iteration " + iteration + " : RMSE = " + Math.sqrt( error/number_of_files__train ) );
	  //System.out.println( Arrays.toString( weights ) );
	  
	  } 
	  while(cost != 0 && iteration<=MAX_ITER);


	}

	static double calculateHypothesis( double[] theta, double[][] feature_matrix, int file_index, int globo_dict_size )
	{
		double hypothesis = 0.0;

		 for (int i = 0; i < globo_dict_size; i++) 
		 {
			 hypothesis += ( theta[i] * feature_matrix[file_index][i] );
		 }
		 //bias
		 hypothesis += theta[ globo_dict_size ];

		 return hypothesis;
	}

	static double[] calculateGradent( double theta[], double[][] feature_matrix, int file_index, int globo_dict_size, double cost, int number_of_files__train)
	{
		double m = number_of_files__train;

		double[] gradient = new double[ globo_dict_size];//one for bias?
		
		for (int i = 0; i < gradient.length; i++) 
		{
			gradient[i] = (1.0/m) * cost * feature_matrix[ file_index ][ i ] ;
		}
		
		return gradient;
	}
	 
	
	public static double randomNumber(int min , int max) 
	{
		DecimalFormat df = new DecimalFormat("#.####");
		double d = min + Math.random() * (max - min);
		String s = df.format(d);
		double x = Double.parseDouble(s);
		return x;
	 } 
	 
	 

}






