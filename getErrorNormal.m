function [err, R, predicted, model] = getErrorNormal(data, crossIndex, para, algoMode)
   %% 计算常规模型测试集误差
    % 输入：
    %   data                - 数据集（算法要求一行是一个样本）
    %   crossIndex          - 当前交叉验证的折
    %   para                - 列向量，优化算法推荐的SVM参数
    % 输出：
    %   err                 - 算法MSE
    %   R                   - 算法R
    %   predicted           - 预测值（只有一个目标值，行向量）
    %   model               - 最终预测时用的模型
    
    crossRow = data.crossRow{crossIndex};
%     crossColumn = data.crossColumn{crossIndex};
    switch algoMode
        case 'SVM'
            model = svmtrain(crossRow.trainY, crossRow.trainX,...                       % -s 3：选择epsilon-SVR模型
                ['-s 3 -c ', num2str(2^para(1)), ' -g ', num2str(2^para(2)) '-q']);     % -q  ：表示关闭控制台输出，然而并没有用
        case 'DT'
            model = fitrtree(crossRow.trainX, crossRow.trainY, 'MinLeafSize', para(1), 'MinParentSize', para(2));
        case 'RF'
            model = TreeBagger(para(1), crossRow.trainX, crossRow.trainY,...
                'MinLeafSize',para(2), 'Method', 'regression');                         % 其他可选，NumPredictorsToSample随机选择的指标数
        case 'KNN'
            model = fitcknn(crossRow.trainX, crossRow.trainY, 'NumNeighbors', para);
        case 'LR'
            [temp, PS] = mapminmax(crossRow.trainY', 0, 1); % 对trainY归一化
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
            temp = mapminmax('reverse', predicted', PS);        % 反归一化回原值范围
            predicted = temp';
        otherwise
            predicted = predict(model, crossRow.testX);
    end
    err = mse(crossRow.testY, predicted);
    R = min(min(corrcoef(crossRow.testY, predicted)));
    predicted = predicted';
    
    %% SVM模型参数说明
    %   cmd
    %       -s svm类型，默认0
    %           0 - C-SVC（多类分类）
    %           1 - v-SVC（多类分类） 
    %           2 - 类SVM
    %           3 - e-SVR
    %           4 - v-SVR
    %       -t 核函数类型，默认2
    %           0 - 线性：u'v
    %           1 - 多项式：(r*u'v + coef0)^degree
    %           2 - RBF函数：exp(-gamma|u-v|^2)
    %           3 - sigmoid：tanh(r*u'v + coef0)
    %       -d degree：核函数中的degree设置(针对多项式核函数)(默认3)
    %       -g r(gamma)：核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)(默认1/k)
    %       -r coef0：核函数中的coef0设置(针对多项式/sigmoid核函数)(默认0)
    %       -c cost：设置C-SVC，e -SVR和v-SVR的参数(损失函数)(默认1)
    %       -n nu：设置v-SVC，一类SVM和v-SVR的参数(默认0.5)
    %       -p p：设置e-SVR中损失函数p的值(默认0.1)
    %       -m cachesize：设置cache内存大小，以MB为单位(默认40)
    %       -e eps：设置允许的终止判据(默认0.001)
    %       -h shrinking：是否使用启发式，0或1(默认1)
    %       -wi weight：设置第几类的参数C为weight*C(C-SVC中的C)(默认1)
    %       -v n：n-fold交互检验模式，n为fold的个数，必须大于等于2
    %       -q：安静模式，即控制台不输出
    %   model
    %       .Parameters     - 参数
    %           .nr_class	- 类的数目
    %           .totalSV	- 总的支持向量数目
    %           .rho        - 判决函数wx+b的b
    %           .Label      - 每个类的标签
    %           .ProbA      - 成对的概率信息，如果b是0,则为空
    %           .ProbB      - 成对的概率信息，如果b是0,则为空
    %           .nSV        - 每个类的支持向量
    %           .sv_coef	- 判决函数的系数
    %           .SVs        - 支持向量