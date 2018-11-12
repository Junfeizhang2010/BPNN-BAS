function [err, R, gap, model] = evaluate(data, para, tuneMode, algoMode, opt)
    %% ����BP������
    % ���룺
    %   data                - ���ݼ�
    %   para                - ���������Ż��㷨�Ƽ���BP������Ȩ�غ�ƫ��
    %   algoMode            - ģ���㷨
    %   isDraw              - �Ƿ�ͼ
    % �����
    %   err                 - ʮ�۽�����֤�Ĳ��Լ�����ֵ
    %   R                   - ʮ�۽�����֤�Ĳ��Լ����ϵ����ֵ
    %   gap                 - Ԥ��ֵ����ʵֵ�Ĳ��������
    %   model               - ����Ԥ��ʱ�õ�ģ��
    
    crossColumn = data.crossColumn;
    %% ѵ�������
    disp(['            �����ڽ���' algoMode 'ģ������...']);
    crossTime = size(crossColumn, 2);
    predicted = [];
    experimental = [];
    for j = 1 : crossTime
        switch algoMode
            case 'BP'
                [err(j), R(j), predictedTemp, model] = getErrorBP(data, j, para, false); % ʹ��ʮ�۽�����֤��ֻ�ܼ���ÿ�εĲ��Լ����
            otherwise
                [err(j), R(j), predictedTemp, model] = getErrorNormal(data, j, para, algoMode); 
        end
        predicted = [predicted, predictedTemp];                 % ������
        experimental = [experimental, crossColumn{j}.testY];    % ������
    end
%     err = mean(err);
%     R = mean(R);
    err = mse(predicted, experimental);
    R = min(min(corrcoef(predicted, experimental)));
    gap = abs(predicted - experimental);
    disp(['              ' algoMode 'ģ��������ϡ�']);
    
    %% ��ͼ
    if ~isempty(opt) || ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        % ��ά�ع�ͼ�����Լ���
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = [algoMode 'Regression of TestSet with Cross'];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż���' algoMode 'ģ��ʮ�۽�����֤���Լ��ع�ͼ'];
        drawRegression(experimental, predicted, figurePara);
        
        % ����Ԥ��ͼ�����Լ���
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = [algoMode 'Predictive of TestSet with Cross'];
        figurePara.label.x = 'Data Number';
        figurePara.label.y2 = 'Residual';
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż���' algoMode 'ģ��ʮ�۽�����֤���Լ�Ԥ��ͼ'];
        figurePara.isSingle = false;
        drawPredicted(experimental, predicted, figurePara);
    end