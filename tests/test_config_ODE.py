Small_orders = list(range(2,11))
High_orders = list(range(10,21))
    
ClassicalImplicitRKMethods = [
    "Gauss"         ,
    "Radau_IA"      ,
    "Radau_IIA"     ,
    "Radau_IB"      ,
    "Radau_IIB"     ,
    "Lobatto_IIIA"  ,
    "Lobatto_IIIB"  ,
    "Lobatto_IIIC"  ,
    "Lobatto_IIIC*" ,
    "Lobatto_IIID"  ,            
    "Lobatto_IIIS"  ,     
]

SymplecticImplicitRKMethodPairs = [
    ("Gauss"        , "Gauss"           ),
    ("Radau_IB"     , "Radau_IB"        ),
    ("Radau_IIB"    , "Radau_IIB"       ),
    ("Lobatto_IIID" , "Lobatto_IIID"    ),
    ("Lobatto_IIIS" , "Lobatto_IIIS"    ),
    ("Lobatto_IIIA" , "Lobatto_IIIB"    ),
    ("Lobatto_IIIC" , "Lobatto_IIIC*"   ),
]   

SymmetricImplicitRKMethodPairs = [
    ("Gauss"        , "Gauss"           ),
    ("Lobatto_IIIA" , "Lobatto_IIIA"    ),
    ("Radau_IB"     , "Radau_IIB"       ),
    ("Lobatto_IIIS" , "Lobatto_IIIS"    ),
    ("Lobatto_IIID" , "Lobatto_IIID"    ),
]  

SymmetricSymplecticImplicitRKMethodPairs = [
    ("Radau_IA"     , "Radau_IIA"       ),
]   
