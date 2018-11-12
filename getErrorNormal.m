function [err, R, predicted, model] = getErrorNormal(data, crossIndex, para, algoMode)
   %% ���㳣��ģ�Ͳ��Լ����
    % ���룺
    %   data                - ���ݼ����㷨Ҫ��һ����һ��������
    %   crossIndex          - ��ǰ������֤����
    %   para                - ���������Ż��㷨�Ƽ���SVM����
    % �����
    %   err                 - �㷨MSE
    %   R                   - �㷨R
    %   predicted           - Ԥ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %   model               - ����Ԥ��ʱ�õ�ģ��
    
    crossRow = data.crossRow{crossIndex};
%     crossColumn = data.crossColumn{crossIndex};
    switch algoMode
        case 'SVM'
            model = svmtrain(crossRow.trainY, crossRow.trainX,...                       % -s 3��ѡ��epsilon-SVRģ��
                ['-s 3 -c ', num2str(2^para(1)), ' -g ', num2str(2^para(2)) '-q']);     % -q  ����ʾ�رտ���̨�����Ȼ����û����
        case 'DT'
            model = fitrtree(crossRow.trainX, crossRow.trainY, 'MinLeafSize', para(1), 'MinParentSize', para(2));
        case 'RF'
            model = TreeBagger(para(1), crossRow.trainX, crossRow.trainY,...
                'MinLeafSize',para(2), 'Method', 'regression');                         % ������ѡ��NumPredictorsToSample���ѡ���ָ����
        case 'KNN'
            model = fitcknn(crossRow.trainX, crossRow.trainY, 'NumNeighbors', para);
        case 'LR'
            [temp, PS] = mapminmax(crossRow.trainY', 0, 1); % ��trainY��һ��
            crossRow.trainY = temp';
%             model = myLR(crossRow, para);
            model = glmfit(crossRow.trainX, [crossRow.trainY, ones(size(crossRow.trainY))], 'binomial', 'link', 'logit');
    end
    switch algoMode
        case 'SVM'
            predicted = svmpredict(crossRow.testY, crossRow.testX, model);
        case 'LR'
%             predicted = 1 ./ (1 + exp(-[ones(size(crossRow.testX, 1), 1), crossRow.testX] * model));
            predicted = glmval(model, crossRow.testX, 'probit', 'size', ones(size(crossRow.testY)));
            temp = mapminmax('reverse', predicted', PS);        % ����һ����ԭֵ��Χ
            predicted = temp';
        otherwise
            predicted = predict(model, crossRow.testX);
    end
    err = mse(crossRow.testY, predicted);
    R = min(min(corrcoef(crossRow.testY, predicted)));
    predicted = predicted';
    
    %% SVMģ�Ͳ���˵��
    %   cmd
    %       -s svm���ͣ�Ĭ��0
    %           0 - C-SVC��������ࣩ
    %           1 - v-SVC��������ࣩ 
    %           2 - ��SVM
    %           3 - e-SVR
    %           4 - v-SVR
    %       -t �˺������ͣ�Ĭ��2
    %           0 - ���ԣ�u'v
    %           1 - ����ʽ��(r*u'v + coef0)^degree
    %           2 - RBF������exp(-gamma|u-v|^2)
    %           3 - sigmoid��tanh(r*u'v + coef0)
    %       -d degree���˺����е�degree����(��Զ���ʽ�˺���)(Ĭ��3)
    %       -g r(gamma)���˺����е�gamma��������(��Զ���ʽ/rbf/sigmoid�˺���)(Ĭ��1/k)
    %       -r coef0���˺����е�coef0����(��Զ���ʽ/sigmoid�˺���)(Ĭ��0)
    %       -c cost������C-SVC��e -SVR��v-SVR�Ĳ���(��ʧ����)(Ĭ��1)
    %       -n nu������v-SVC��һ��SVM��v-SVR�Ĳ���(Ĭ��0.5)
    %       -p p������e-SVR����ʧ����p��ֵ(Ĭ��0.1)
    %       -m cachesize������cache�ڴ��С����MBΪ��λ(Ĭ��40)
    %       -e eps�������������ֹ�о�(Ĭ��0.001)
    %       -h shrinking���Ƿ�ʹ������ʽ��0��1(Ĭ��1)
    %       -wi weight�����õڼ���Ĳ���CΪweight*C(C-SVC�е�C)(Ĭ��1)
    %       -v n��n-fold��������ģʽ��nΪfold�ĸ�����������ڵ���2
    %       -q������ģʽ��������̨�����
    %   model
    %       .Parameters     - ����
    %           .nr_class	- �����Ŀ
    %           .totalSV	- �ܵ�֧��������Ŀ
    %           .rho        - �о�����wx+b��b
    %           .Label      - ÿ����ı�ǩ
    %           .ProbA      - �ɶԵĸ�����Ϣ�����b��0,��Ϊ��
    %           .ProbB      - �ɶԵĸ�����Ϣ�����b��0,��Ϊ��
    %           .nSV        - ÿ�����֧������
    %           .sv_coef	- �о�������ϵ��
    %           .SVs        - ֧������