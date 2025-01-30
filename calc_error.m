function [mae,rmse,mape,error]=calc_error(x1,x2)

error=x2-x1;  
rmse=sqrt(mean(error.^2));
disp(['1.MSE',num2str(mse(x1-x2))])
disp(['2.RMSE',num2str(rmse)])

 mae=mean(abs(error));
disp(['3.MAE£º',num2str(mae)])

 mape=mean(abs(error)/x1);
 disp(['4.MAPE£º',num2str(mape*100),'%'])
Rsq1 = 1 - sum((x1 - x2).^2)/sum((x1 - mean(x2)).^2);
disp(['5.R2£º',num2str(Rsq1),'%'])
end

