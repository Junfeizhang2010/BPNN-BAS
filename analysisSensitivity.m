function analysisSensitivity(data, model, tuneMode, algoMode, opt)
    %% �����Է���
    % ���룺
    %   data    - ����
    %   model   - ѵ���õ�ģ��
    
    disp(['            �����ڷ�������ָ���' algoMode 'ģ�͵�������...']);
    all = data.all;
    %% �����ֵ����ʽ����y/y
    n = data.inputDimCount;
    for i = 1 : n
        l = min(all(:,i));
        h = max(all(:,i));
        unit{i} = linspace(l, h * 0.9, 10)';
    end  
    x1 = eachBlock(n, unit);  % ÿ��һ������
    switch algoMode
        case 'BP'
            predicted1 = sim(model, x1');  % ������
        case 'SVM'
            predicted1 = svmpredict(rand(size(x1, 1), 1), x1, model);  % �㷨Ҫ�����������ʵ�Ľ�������ڼ���MSE���˴�����Ҫ����������뼴��
        case 'LR'
            predicted1 = 1 ./ (1 + exp([ones(size(x1, 1), 1), -x1] * model));
        otherwise
            predicted1 = predict(model, x1);
    end
    for i = 1 : n
        x2 = [x1(:,i) * 1.1, x1(:, [1 : i - 1 ,i + 1 : end])];
        switch algoMode
            case 'BP'
                predicted2 = sim(model, x2');  % ������
            case 'SVM'
                predicted2 = svmpredict(rand(size(x2, 1), 1), x2, model);
            case 'LR'
                predicted2 = 1 ./ (1 + exp([ones(size(x2, 1), 1), -x2] * model));
            otherwise
                predicted2 = predict(model, x2);
        end
        score(i) = mean(mean((predicted2 - predicted1) ./ predicted1)); % ��ֹ�ʽ
    end
    [~, index] = sort(abs(score), 'descend');
    temp = score(index); % �˴�������temp����Ϊ����Ļ�ͼ����Ҫ��score���ܱ����򣬷����������еı�ǩ����Ӧ
    figurePara = setFigurePara(data.dataIndex, data.posOutput);
    fprintf(['              ����ָ���' algoMode 'ģ�͵������Է�����ϣ�ָ�������Է�������Ϊ��']);
    for i = 1 : size(temp, 2)
        fprintf([figurePara.label.input{index(i)}, ' �� ' ,num2str(temp(i))]);
        if i ~= size(temp, 2)
            fprintf('��');
        else
            disp('��');
        end
    end
    
    %% ��ͼ
    if ~isempty(opt) || ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = ['Sensitivity Analysis Tornato Graph of Influential Variable with ' algoMode];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż���' algoMode 'ģ��ָ�������Է��������ͼ'];
        figurePara.label.y = 'Importance score';
        figurePara.label.x = 'Influential variable';
        drawTornado(score, figurePara); % score������ǰ���򣬷��������ݱ�ǩ����Ӧ
    end
    
function block = eachBlock(i, unit)
    if i == 1
        block = unit{1};
    else
        block = eachBlock(i - 1, unit);
        block = repmat(block, 10, 1);
        block = cat(2, block, reshape(repmat(unit{i}', 10^(i-1), 1), 10^i, 1));
    end