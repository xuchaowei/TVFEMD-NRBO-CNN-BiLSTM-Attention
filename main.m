%% 初始化
clear
close all
clc
warning off

%%% 导入数据
X_imf = xlsread('数据集2.xlsx');
num_samples = length(X_imf);       % 样本个数 
kim = 7;                      % 延时步长（kim个历史数据作为自变量） 时间步长
zim = 1;                      % 跨zim个时间点进行预测  预见期--降雨径流必须有可以不带径流这一列特征
or_dim = size(X_imf,2);  %所有输入特征包括目标变量
%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end

%%  数据分析
num_size = 0.5;                              % 训练集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);   %350个训练的  总共488个数据集  0.7

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);  %138个测试的  总共488个数据集 0.3

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train,0,1);
p_test = mapminmax('apply',P_test,ps_input);

[t_train, ps_output] = mapminmax(T_train,0,1);
t_test = mapminmax('apply',T_test,ps_output);

%%  数据平铺---就是组合一下，把第一个数据集组合到一个矩阵里，平铺操作是为了确保原始数据能够与神经网络模型的输入要求相匹配
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致--为了与网络输出层相一致
p_train =  double(reshape(p_train, f_, 1, 1, M));
p_test  =  double(reshape(p_test , f_, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

%%  数据格式转换--也是和集组合到一个矩阵里，确保原始数据能够与神经网络模型的输入要求相匹配
% 从四维数组 p_train 和 p_test 中提取出一个样本，并将其存储到单元格数组 Lp_train 和 Lp_test 中。
% 总之，这段代码的目的是将原始数据转换成特定格式的单元格数组，以便后续的数据处理和分析。
for i = 1 : M
    Lp_train{i, 1} = p_train(:, :, 1, i);
end

for i = 1 : N
    Lp_test{i, 1}  = p_test( :, :, 1, i);
end
%%  优化算法参数设置--是神经网络结构的参数-比如学习率、隐藏层节点数、正则化系数系数等
% 文章：The proposed NRBO is basically developed using the Newton-Raphson method to investigate the search space’s best positions 
% and find the best solution.
% SearchAgents_no = 10;                  % 数量  种群中的粒子数--猴子
% Max_iteration = 2;                    % 最大迭代次数
% dim = 3;                               % 优化参数个数
% lb = [1e-3,10 1e-4];                 % 参数取值下界(学习率，隐藏层节点，正则化系数)
% ub = [1e-2, 30,1e-1];                 % 参数取值上界(学习率，隐藏层节点，正则化系数)

% tic; % 开始计时
% fitness = @(x)fical(x,Lp_train,t_train,ps_output,T_train,f_);
% [Best_score,Best_pos,curve]=NRBO(SearchAgents_no,Max_iteration,lb ,ub,dim,fitness) %耗时的地方
% elapsedTime = toc; % 停止计时并获取耗时时间
% disp(['NRBO-CNN-BiLSTM-Attention优化参数执行时间: ', num2str(elapsedTime), ' 秒']);

% best_hd  = round(Best_pos(1, 2)); % 最佳隐藏层节点数
% best_lr= Best_pos(1, 1);% 最佳初始学习率
% best_l2 = Best_pos(1, 3);% 最佳L2正则化系数
% Best_pos(1, 2) = round(Best_pos(1, 2));     

best_hd  = 57;
best_lr = 0.000393569777940371;
best_l2 = 0.00001;

%%  建立模型
lgraph = layerGraph();                                                 % 建立空白网络结构

tempLayers = [
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 32
lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
 
tempLayers = [
    reluLayer("Name", "relu_1")                                        % 激活层
    convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[3, 1] 步长[1, 1] 通道数 64
    reluLayer("Name", "relu_2")];                                      % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
    fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
    reluLayer("Name", "relu_3")                                        % 激活层
    fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
    sigmoidLayer("Name", "sigmoid")];                                  % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
    bilstmLayer(best_hd, "Name", "bilstm", "OutputMode", "last")                 % BiLSTM层
    fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
    regressionLayer("Name", "regressionoutput")];                      % 回归层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % 折叠层输出 连接 反折叠层输入  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出

%% Set the hyper parameters for unet training
options = trainingOptions('adam', ...                 % 浼樺寲绠楁硶Adam
    'MiniBatchSize',128, ...    
    'MaxEpochs', 30, ...                            % 鏈€澶ц缁冩鏁?
    'GradientThreshold', 1, ...                       % 姊害闃堝€?
    'InitialLearnRate', best_lr, ...         % 鍒濆瀛︿範鐜?
    'LearnRateSchedule', 'piecewise', ...             % 瀛︿範鐜囪皟鏁?
    'LearnRateDropPeriod',15, ...                   % 璁粌100娆″悗寮€濮嬭皟鏁村涔犵巼
    'LearnRateDropFactor',0.01, ...                    % 瀛︿範鐜囪皟鏁村洜瀛?
    'L2Regularization', best_l2, ...         % 姝ｅ垯鍖栧弬鏁?
    'ExecutionEnvironment', 'gpu',...                 % 璁粌鐜
    'Verbose', false, ...                                 % 鍏抽棴浼樺寲杩囩▼
    'Plots', 'training-progress', ...
     ValidationData={Lp_test,t_test}, ...
     ValidationFrequency=1, ...
    ObjectiveMetricName="loss");                    % 鐢诲嚭鏇茬嚎
%%  训练模型
net = trainNetwork(Lp_train, t_train, lgraph, options);

%%  模型预测
t_sim1 = predict(net, Lp_train);
t_sim2 = predict(net, Lp_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1', ps_output);
T_sim2 = mapminmax('reverse', t_sim2', ps_output);
T_sim1=double(T_sim1);
T_sim2=double(T_sim2);



[kge,nse,re,r2] = calcansekge(T_train,T_sim1);
fprintf('\n')


[kge,nse,re,r2] = calcansekge(T_test,T_sim2);
fprintf('\n')

ResultTimeSeriTrain = [T_train' T_sim1'];
ResultTimeSeriTest = [T_test' T_sim2'];
