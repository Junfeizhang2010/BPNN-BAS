function untunedCompare(data, tunedErr, tunedR, tuneMode, algoMode, opt)
    %% �Ƚ�δ���Ż����
     % ���룺
    %   data      - ���ݼ�
    %   tunedErr  - ʮ�۽�����֤�Ĳ��Լ�����ֵ
    %   tunedR    - ʮ�۽�����֤�Ĳ��Լ����ϵ����ֵ
    %   tuneMode  - �Ż��㷨
    %   algoMode  - ģ���㷨
    %   opt       - �㷨����
    
    %% ��������������������*10����
    disp(['            ��������֤���Ż���' algoMode 'ģ����δ���Ż��Ĳ���...']);
    fprintf('                  ');
    time = size(data.crossColumn, 2);
    if isfield(opt, 'compare') && isfield(opt.compare, 'pointCount')
        multiple = floor(opt.compare.pointCount / time);
    else
        multiple = 10;
    end
    
    for j = 1 : time
        for i = 0 : multiple
            if i * 10 + j > opt.compare.pointCount, j = time; break; end
            switch algoMode
                case 'BP'
                    para.neuronCount = ceil(rand(1,ceil(rand(1) * 4)) * 60);
                    [err(i * 10 + j), R(i * 10 + j)] = getErrorBP(data, j, para, []);
                case 'SVM'
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, rands(1, 2) * 7 - 3, algoMode);
                case {'DT', 'RF'}
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, round(rand(1, 2) * 100 + 1), algoMode);
                case 'KNN'
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, round(rand(1) * 50 + 1), algoMode);
                case 'LR'
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, rand(data.inputDimCount + 1, 1) * 5, algoMode);
            end
        end
        fprintf(['�����' num2str(10 * j) '�β���']);
        if j ~= time
            fprintf('��');
            if rem(j, 5) == 0, fprintf('\n                  '); end
        else
            fprintf('��\n');
        end
    end
    disp(['              ��֤���Ż���' algoMode 'ģ����δ���Ż��Ĳ�����ϣ��Ż���'...
        algoMode 'ģ�ͳ���' num2str(length(err(err >= tunedErr)) / length(err) * 100) '%��δ�Ż�ģ�͡�']);
    
    %% ��ͼ
    if ~isempty(opt) ||~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        % MSE�Ż��Ա�ͼ
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = ['Tune MSE and untune MSE of TrainSet Predictive with' algoMode];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż���' algoMode 'ģ���Ż�Ч�����Ա�ͼ'];
        figurePara.label.y = figurePara.label.mse;
        drawTunedAndUntuned(tunedErr, err, figurePara);
        
        % R�Ż��Ա�ͼ
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = ['Tune MSE and untune MSE of TrainSet Predictive with ' algoMode];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż���' algoMode 'ģ���Ż�Ч�����ϵ���Ա�ͼ'];
        figurePara.label.y = 'R';
        drawTunedAndUntuned(tunedR, R, figurePara);
    end