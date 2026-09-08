from MyClasses.cls_CalculateMetrics import cls_CalculateMetrics



def fn_ShowCfMatrixDetails(cf_matrix):
    ## obj
    calcmet = cls_CalculateMetrics(cf_matrix)

    precision = calcmet.calc_precision()
    precision = [round(elem, 2) * 100 for elem in precision]
    #precision = [round(elem * 100) for elem in precision]

    recall = calcmet.calc_recall()
    recall = [round(elem, 2) * 100 for elem in recall]
    #recall = [round(elem * 100) for elem in recall]

    f1_score = calcmet.calc_f1_score()
    f1_score = [round(elem, 2) * 100 for elem in f1_score]
    #f1_score = [round(elem * 100) for elem in f1_score]

    acc = calcmet.calc_accuracy()
    acc = round(acc * 100)

    # ----- cf_matrix
    print("-" * 15)
    print("Confusion Matrics: ")
    print(cf_matrix)   

    # ----- Print results
    print()
    print("-" * 15)
    print("accuracy  :", acc)
    print("precision :", precision)
    print("recall    :", recall)
    print("f1_score  :", f1_score)
    

    return