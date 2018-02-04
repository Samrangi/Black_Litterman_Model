import csv
import sys
import re
import numpy as np
import pandas as pd
from pandas import DataFrame
import statistics
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


def black_litterman_model(tau_omega, P, Q):
	#data = pd.read_csv('trial1_input_data.csv',index_col = 'YearM', parse_dates = True)
	with open("Black_Litterman_Output.txt","a") as op:
		op.write("\n********************BEGINNING OF SECTION********************\n");

		#Reading the Data
		data = pd.read_csv('input_Data.csv',index_col = 'YearM', parse_dates = True)
		# data.to_csv('Black_Litterman_Output.txt', header=True, index=False, sep='\t', mode='a')
		
		#Calculating the Covariance among the three asset classes
		cov_mat = data.cov();
		# print("\nCovariance Matrix: \n", cov_mat);
		# op.write("\nCovariance Matrix:\n");
		# print(f"""{cov_mat}""", file = op);

		#Calculating the correlation among the three asset classes
		corr_mat = data.corr();
		# print("\nCorrelation Matrix: \n", corr_mat);
		# op.write("\nCorrelation Matrix:\n");
		# print(f"""{corr_mat}""", file = op);

		#Risk Aversion Parameter
		risk_ave = 3;
		op.write("\nRisk Aversion:\n");
		print(f"""{risk_ave}""", file = op);

		#Scalar on the Uncertainity in the Prior
		tau = 0.10;
		op.write("\nScalar on the Uncertainity in the Prior (Tau):\n");
		print(f"""{tau}""", file = op);

		#Market Weights of the three asset classes in order
		mkt_wts = [0.5, 0.4, 0.1];
		op.write("\nMarket weights(Prior):\n");
		print(f"""{mkt_wts}""", file = op);

		#Calculating the Prior Returns
		prior_ret = risk_ave * np.dot(cov_mat, mkt_wts);
		# op.write("\nPrior Returns:\n");
		# print(f"""{prior_ret}""", file = op);
		# print("\nPrior Returns: \n", prior_ret);
		
		#Calculating the Market Variance
		mkt_var = np.dot(np.transpose(mkt_wts), np.dot(cov_mat, mkt_wts));
		# op.write("\nMarket Variance:\n");
		# print(f"""{mkt_var}""", file = op);
		# print("\nMarket Variance: \n", mkt_var);

		#Calculating the expected market excess return
		mkt_exp_exc_ret = risk_ave* mkt_var;
		# op.write("\nMarket Expected Excess Return:\n");
		# print(f"""{mkt_exp_exc_ret}""", file = op);
		# print("\nMarket Expected Excess Return: \n ", mkt_exp_exc_ret);

		#Calculating the standard deviation of the market
		mkt_std_dev = np.sqrt(np.dot(np.dot(np.transpose(mkt_wts), cov_mat), mkt_wts));
		# op.write("\nMarket Expected Excess Return:\n");
		# print(f"""{mkt_std_dev}""", file = op);
		# print("\nMarket Standard Deviation: \n", mkt_std_dev);

		#Calculating the sharpe ratio
		sharpe_ratio = mkt_exp_exc_ret/mkt_std_dev;
		# op.write("\nSharpe Ratio:\n");
		# print(f"""{sharpe_ratio}""", file = op);
		# print("\nSharpe Ratio: \n", sharpe_ratio);


		# P = np.zeros((2,), dtype=[('US Equity', 'f4'),('Foreign Equity', 'f4'),('Emerging Equity', 'f4')]);
		# P = pd.DataFrame(P, index=['View 1', 'View 2']);
		# P[:] = [(1.5,0,0),(0,1,-1)];
		
		#Writing the P matrix
		op.write("\nP matrix:\n");
		print(f"""{P}""", file = op);
		print("\nP Matrix: \n", P);


		# Q = [0.015, 0.030];

		#Writing the Q matrix
		op.write("\nQ matrix:\n");
		print(f"""{Q}""", file = op);
		print("\nQ matrix: \n", Q);


		# tau_omega = 0.10;
		omega = tau_omega * np.dot(P,np.dot(cov_mat,np.transpose(P)));
		op.write("\nScalar on Manager's Views (Omega) :\n");
		print(f"""{omega}""", file = op);
		print("\nOmega: \n", omega);


		#Calculating the prior precision views
		prior_precision_views = np.dot(P,np.dot(tau*cov_mat,np.transpose(P)));
		op.write("\nPrior Precision Views :\n");
		print(f"""{prior_precision_views}""", file = op);
		print("\nPrior Precision of Views: \n", prior_precision_views);


		#Covariance times Tau
		cov_times_tau = cov_mat* tau;
		# op.write("\nCovariance Times Tau :\n");
		# print(f"""{cov_times_tau}""", file = op);
		# print("\nCovariance times Tau: \n", prior_precision_views);
		

		#Calculating the posterior Returns
		posterior_ret = prior_ret + np.dot(np.dot(cov_times_tau,np.dot(np.transpose(P),np.linalg.inv(omega+prior_precision_views))),Q - np.dot(P,prior_ret));
		op.write("\nPosterior returns :\n");
		print(f"""{posterior_ret}""", file = op);
		print("\nPosterior Returns: \n", posterior_ret);

		#Calculating the posterior Return Distribution
		posterior_ret_dist = cov_mat + cov_times_tau - np.dot(np.dot(np.dot(cov_times_tau,np.transpose(P)),np.linalg.inv(omega+prior_precision_views)),np.dot(P,cov_times_tau));
		op.write("\nPosterior Return Distribution :\n");
		print(f"""{posterior_ret_dist}""", file = op);
		print("\nPosterior Return Distribution: \n", posterior_ret_dist);
		

		#Calculating the Unconstrained Optimum weights
		optima_unconst = np.dot(np.transpose(posterior_ret),np.linalg.inv(risk_ave * posterior_ret_dist))
		op.write("\nUnconstrained Optimal Weights :\n");
		print(f"""{optima_unconst}""", file = op);
		print("\nUnconstrained Optimal Weights: \n", optima_unconst);


		#Calculating the Normalised Optimum weights
		optima_unconst_norm = optima_unconst/sum(optima_unconst);
		op.write("\nNormalised Optimal Weights :\n");
		print(f"""{optima_unconst_norm}""", file = op);
		print("\nNormalised Optimal Wieghts: \n", optima_unconst_norm);


		#Calculating the Optimum Expected Returns
		optima_exp_ret =np.dot(np.transpose(posterior_ret),optima_unconst_norm);
		op.write("\nOptimum Expected Returns :\n");
		print(f"""{optima_exp_ret}""", file = op);
		print("\nOptimum Expected Returns: \n", optima_exp_ret);

		#Calculating the Optimum Variance
		optima_var = np.dot(np.dot(np.transpose(optima_unconst_norm), posterior_ret_dist),optima_unconst_norm);
		op.write("\nOptimum Variance :\n");
		print(f"""{optima_var}""", file = op);
		print("\nOptimum Variance: \n",optima_var);
		
		#Calculating the Optimum Standard Deviation
		optima_std_dev = np.sqrt(optima_var);
		op.write("\nOptimum Standard Deviation :\n");
		print(f"""{optima_std_dev}""", file = op);
		print("\nOptimum Standard Deviation: \n", optima_std_dev);


		#Calculating the Optimum Standard Deviation
		optima_sharpe_ratio = optima_exp_ret/optima_std_dev;
		op.write("\nOptimum Sharpe Ratio :\n");
		print(f"""{optima_sharpe_ratio}""", file = op);
		print("\nOptimum Sharpe Ratio: \n",optima_sharpe_ratio);

		op.write("\n***********************END OF SECTION***********************\n");
		print("\n********************END OF SECTION********************\n")
	return 0;



if __name__=='__main__':
	tau_omega_1 = 0.10;
	tau_omega_2 = 0.01;
	tau_omega_3 = 0.10;

	P_1 = np.zeros((2,), dtype=[('US Equity', 'f4'),('Foreign Equity', 'f4'),('Emerging Equity', 'f4')]);
	P_1 = pd.DataFrame(P_1, index=['View 1', 'View 2']);
	P_1[:] = [(1.5,0,0),(0,1,-1)];

	P_2 = np.zeros((2,), dtype=[('US Equity', 'f4'),('Foreign Equity', 'f4'),('Emerging Equity', 'f4')]);
	P_2 = pd.DataFrame(P_2, index=['View 1', 'View 2']);
	P_2[:] = [(1.5,0,0),(0,1,-1)];

	P_3 = np.zeros((2,), dtype=[('US Equity', 'f4'),('Foreign Equity', 'f4'),('Emerging Equity', 'f4')]);
	P_3 = pd.DataFrame(P_3, index=['View 1', 'View 2']);
	P_3[:] = [(1,-1,0),(0,0,1.5)];

	Q_1 = [0.015, 0.030];

	Q_2 = [0.015, 0.030];

	Q_3 = [0.020, 0.015];

	black_litterman_model(tau_omega_1, P_1, Q_1);
	black_litterman_model(tau_omega_2, P_2, Q_2);
	black_litterman_model(tau_omega_3, P_3, Q_3);
		