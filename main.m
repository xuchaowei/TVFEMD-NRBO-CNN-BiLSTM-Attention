%% ��ʼ��
clear
close all
clc
warning off

%%% ��������
X_imf = xlsread('���ݼ�2.xlsx');
num_samples = length(X_imf);       % �������� 
kim = 7;                      % ��ʱ������kim����ʷ������Ϊ�Ա����� ʱ�䲽��
zim = 1;                      % ��zim��ʱ������Ԥ��  Ԥ����--���꾶�������п��Բ���������һ������
or_dim = size(X_imf,2);  %����������������Ŀ�����
%  �ع����ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end

%%  ���ݷ���
num_size = 0.5;                              % ѵ����ռ���ݼ�����
outdim = 1;                                  % ���һ��Ϊ���
num_samples = size(res, 1);                  % ��������
res = res(randperm(num_samples), :);         % �������ݼ�����ϣ������ʱ��ע�͸��У�
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��

%%  ����ѵ�����Ͳ��Լ�
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);   %350��ѵ����  �ܹ�488�����ݼ�  0.7

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);  %138�����Ե�  �ܹ�488�����ݼ� 0.3

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train,0,1);
p_test = mapminmax('apply',P_test,ps_input);

[t_train, ps_output] = mapminmax(T_train,0,1);
t_test = mapminmax('apply',T_test,ps_output);

%%  ����ƽ��---�������һ�£��ѵ�һ�����ݼ���ϵ�һ�������ƽ�̲�����Ϊ��ȷ��ԭʼ�����ܹ���������ģ�͵�����Ҫ����ƥ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��--Ϊ���������������һ��
p_train =  double(reshape(p_train, f_, 1, 1, M));
p_test  =  double(reshape(p_test , f_, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

%%  ���ݸ�ʽת��--Ҳ�Ǻͼ���ϵ�һ�������ȷ��ԭʼ�����ܹ���������ģ�͵�����Ҫ����ƥ��
% ����ά���� p_train �� p_test ����ȡ��һ��������������洢����Ԫ������ Lp_train �� Lp_test �С�
% ��֮����δ����Ŀ���ǽ�ԭʼ����ת�����ض���ʽ�ĵ�Ԫ�����飬�Ա���������ݴ���ͷ�����
for i = 1 : M
    Lp_train{i, 1} = p_train(:, :, 1, i);
end

for i = 1 : N
    Lp_test{i, 1}  = p_test( :, :, 1, i);
end
%%  �Ż��㷨��������--��������ṹ�Ĳ���-����ѧϰ�ʡ����ز�ڵ���������ϵ��ϵ����
% ���£�The proposed NRBO is basically developed using the Newton-Raphson method to investigate the search space��s best positions 
% and find the best solution.
% SearchAgents_no = 10;                  % ����  ��Ⱥ�е�������--����
% Max_iteration = 2;                    % ����������
% dim = 3;                               % �Ż���������
% lb = [1e-3,10 1e-4];                 % ����ȡֵ�½�(ѧϰ�ʣ����ز�ڵ㣬����ϵ��)
% ub = [1e-2, 30,1e-1];                 % ����ȡֵ�Ͻ�(ѧϰ�ʣ����ز�ڵ㣬����ϵ��)

% tic; % ��ʼ��ʱ
% fitness = @(x)fical(x,Lp_train,t_train,ps_output,T_train,f_);
% [Best_score,Best_pos,curve]=NRBO(SearchAgents_no,Max_iteration,lb ,ub,dim,fitness) %��ʱ�ĵط�
% elapsedTime = toc; % ֹͣ��ʱ����ȡ��ʱʱ��
% disp(['NRBO-CNN-BiLSTM-Attention�Ż�����ִ��ʱ��: ', num2str(elapsedTime), ' ��']);

% best_hd  = round(Best_pos(1, 2)); % ������ز�ڵ���
% best_lr= Best_pos(1, 1);% ��ѳ�ʼѧϰ��
% best_l2 = Best_pos(1, 3);% ���L2����ϵ��
% Best_pos(1, 2) = round(Best_pos(1, 2));     

best_hd  = 57;
best_lr = 0.000393569777940371;
best_l2 = 0.00001;

%%  ����ģ��
lgraph = layerGraph();                                                 % �����հ�����ṹ

tempLayers = [
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % ��������㣬�������ݽṹΪ[f_, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % ���������۵���
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % ����� �����[3, 1] ����[1, 1] ͨ���� 32
lgraph = addLayers(lgraph,tempLayers);                                 % ����������ṹ����հ׽ṹ��
 
tempLayers = [
    reluLayer("Name", "relu_1")                                        % �����
    convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % ����� �����[3, 1] ����[1, 1] ͨ���� 64
    reluLayer("Name", "relu_2")];                                      % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % ȫ��ƽ���ػ���
    fullyConnectedLayer(16, "Name", "fc_2")                            % SEע�������ƣ�ͨ������1 / 4
    reluLayer("Name", "relu_3")                                        % �����
    fullyConnectedLayer(64, "Name", "fc_3")                            % SEע�������ƣ���Ŀ��ͨ������ͬ
    sigmoidLayer("Name", "sigmoid")];                                  % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % ��˵�ע����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % �������з��۵���
    flattenLayer("Name", "flatten")                                    % ������ƽ��
    bilstmLayer(best_hd, "Name", "bilstm", "OutputMode", "last")                 % BiLSTM��
    fullyConnectedLayer(1, "Name", "fc")                               % ȫ���Ӳ�
    regressionLayer("Name", "regressionoutput")];                      % �ع��
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % �۵������ ���� ���������;
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % �۵������ ���� ���۵�������  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % �������� ���� �����
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % �������� ���� ȫ��ƽ���ػ�
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % �������� ���� ��˲�
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % ȫ������� ���� ��˲�
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % ������

%% Set the hyper parameters for unet training
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MiniBatchSize',128, ...    
    'MaxEpochs', 30, ...                            % 朢�大训练次�?
    'GradientThreshold', 1, ...                       % 梯度阈��?
    'InitialLearnRate', best_lr, ...         % 初始学习�?
    'LearnRateSchedule', 'piecewise', ...             % 学习率调�?
    'LearnRateDropPeriod',15, ...                   % 训练100次后弢�始调整学习率
    'LearnRateDropFactor',0.01, ...                    % 学习率调整因�?
    'L2Regularization', best_l2, ...         % 正则化参�?
    'ExecutionEnvironment', 'gpu',...                 % 训练环境
    'Verbose', false, ...                                 % 关闭优化过程
    'Plots', 'training-progress', ...
     ValidationData={Lp_test,t_test}, ...
     ValidationFrequency=1, ...
    ObjectiveMetricName="loss");                    % 画出曲线
%%  ѵ��ģ��
net = trainNetwork(Lp_train, t_train, lgraph, options);

%%  ģ��Ԥ��
t_sim1 = predict(net, Lp_train);
t_sim2 = predict(net, Lp_test );

%%  ���ݷ���һ��
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
