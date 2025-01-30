function fitness = fical(x,Lp_train,t_train,ps_output,T_train,f_);
%%  ���������л�ȡѵ������
% ����ֻ��x����������֪�ģ�x����3�����������½磬ÿ�δ�һ��
    Lp_train = evalin('base', 'Lp_train');
    t_train = evalin('base', 't_train');
    ps_output = evalin('base', 'ps_output');
    T_train = evalin('base', 'T_train');
     f_ = evalin('base', 'f_');
    
best_hd  = round(x(1, 2)); % ������ز�ڵ���
best_lr= x(1, 1);% ��ѳ�ʼѧϰ��
best_l2 = x(1, 3);% ���L2����ϵ��
%%  ����ģ��
lgraph = layerGraph();                                                 % �����հ�����ṹ


%�����ѧϰ�У������ĳߴ�ͨ����ʾΪ [height width channels]��������[f_, 1, 1]
% ���� height ������ͼ��ĸ߶ȣ�width �ǿ�ȣ�channels ��ͨ������
% ���ڻҶ�ͼ����ԣ�ͨ��ֻ��һ��ͨ������� channels ��ֵΪ 1������ʱ�����еĻ�����ȵ�ֵΪ1������Ϊ[f_, 1, 1]
%��������ߴ�Ϊ [65 1 1] ��ͼ�����ݣ�sequenceFoldingLayer �Ὣ��ת��Ϊһ�������Ա�����ľ������Դ���
tempLayers = [
    %65*1 1��
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % ��������㣬�������ݽṹΪ[f_, 1, 1]-����13*5=65��ֵ
    sequenceFoldingLayer("Name", "seqfold")];                          % ���������۵��� �����۵������ڽ�������������ת�����ʺϾ���㴦��ĸ�ʽ��
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��
% �ο�ʾ��ͼhttps://blog.51cto.com/u_16099347/10357725
%���Ӿ�����������ܻ���������ı���������ʹ�����ܹ�ѧϰ�����ӵ����������ǣ����������������࣬
% ���ܻᵼ�¹���ϣ��ر��������ݼ���С������¡���ˣ���Ҫ���ݾ����������Ȩ���ѡ��
%����˵�ͨ����Ӧ��������ͨ������ͬ����������ͨ�������ɾ���˵ĸ�������
%��һ����Ҫ���ٸ�featuremap���ͼ�������ٸ������
%�м��������,�ͻ��������Feature map����32��featureͼ
%65-3+1=63  32�ţ���32����Ԫ/������ǲ�ֵͬ��
tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % ����� �����[3, 1-1�������1��һ��] ����[1, 1] ͨ���� 32Ҳ���Ǿ��������32
lgraph = addLayers(lgraph,tempLayers);                                 % ����������ṹ����հ׽ṹ��
 
%Ϊ����������ķ��������������������ı��������ÿ�������󶼻��һ�������,ReLU ��������ı����ݵ���״��
tempLayers = [
    reluLayer("Name", "relu_1")  %���ı�ߴ���������ѧϰ % �����
     %63-3+1=61  64��
    convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % ����� �����[3, 1] ����[1, 1] ͨ���� 64
    reluLayer("Name", "relu_2")];                                      % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��


%%����ȫ���Ӳ㣬�ɽ�������ߴ��ͼ�� https://blog.csdn.net/SmartDemo/article/details/123889624
%ȫ���Ӳ���ʵ����һ��Ȩ�ؾ���-ȫ���Ӳ㽫�ػ�ƽ��ֵ֮���featuremap����ƴ������µģ������ܶ�
tempLayers = [
    % ��61*1��64����ÿ���ľ�ֵ��Ȼ��õ�64�� 1*1���� 
    globalAveragePooling2dLayer("Name", "gapool")                      % ȫ��ƽ���ػ���
    %16����Ԫ��ÿ����ԪȨ�ز�ͬ����64��������㣬�õ�һ��ֵ���ܹ��õ�16��1*1
    fullyConnectedLayer(16, "Name", "fc_2")                            % SEע�������ƣ�ͨ������1 / 4
    reluLayer("Name", "relu_3")                                        % �����
    %64����Ԫ��ÿ����ԪȨ�ز�ͬ����16��������㣬�õ�һ��ֵ���ܹ��õ�64��1*1---����
    fullyConnectedLayer(64, "Name", "fc_3")                            % SEע�������ƣ���Ŀ��ͨ������ͬ--
    %��ÿ��Ԫ�ص�ȡֵ��Χ������ 0 �� 1 ֮�䣬��ʾ��Ӧ��Ԫ�ļ���̶ȡ���ÿ����Ԫ�ļ���̶�
    sigmoidLayer("Name", "sigmoid")];                                  % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

%���ע����
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

%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MaxEpochs', 100, ...                 % ����������
    'InitialLearnRate', best_lr, ...          % ��ʼѧϰ��Ϊ0.01
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1, ...        % ѧϰ���½����� 0.5
     'L2Regularization',best_l2,...
    'LearnRateDropPeriod', 50, ...        % ����700��ѵ���� ѧϰ��Ϊ 0.01 * 0.1
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'Plots', 'none', ...      % ��������
    'Verbose', false);


%% ѵ���������
net = trainNetwork(Lp_train, t_train, lgraph, options);
%% ѵ�����������
t_sim1 = predict(net, Lp_train);
T_sim1 = mapminmax('reverse', t_sim1', ps_output);
%%  ������Ӧ��
fitness = sqrt(sum((T_sim1 - T_train).^2)./length(T_sim1));
disp(['��ʼ Fitness = ' num2str(fitness)]);

end