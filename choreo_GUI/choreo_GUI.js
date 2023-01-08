
function handleMessageFromWorker(message) {

    if ((typeof message.data.funname != "undefined") && (typeof message.data.args != "undefined")) {

        // console.log("Attempting to execute function",message.data.funname,"with arguments",message.data.args);

        window[message.data.funname](message.data.args);

    } else {

        // console.log('Main thread could not resolve message from worker :',message);
        // console.log(message.data);

    }

}

var DownloadTxtFile = (function () {
    var a = document.createElement("a");
    document.body.appendChild(a);
    a.style = "display: none";
    return function (args) {
        blob = new Blob([args.data],{type : 'application/text'})
        url = window.URL.createObjectURL(blob);
        a.href = url;
        a.download = args.filename;
        a.click();
        window.URL.revokeObjectURL(url);
    };
}());

async function Play_Loop_From_Python(args){

	var displayCanvas = document.getElementById("displayCanvas")

    var event = new Event('StopAnimationFromOutsideCanvas')
    displayCanvas.dispatchEvent(event)

    SolName = args.solname
    var txt = await args.JSON_data.text()
    PlotInfo = JSON.parse(txt)
    UpdateNowPlaying()
    Pos = {"data":args.NPY_data,"shape":args.NPY_shape}

    var event = new Event('EnableAnimationFromOutsideCanvas')
    displayCanvas.dispatchEvent(event)


    if (document.getElementById('checkbox_DisplayLoopsDuringSearch').checked) {

        trajectoriesOn = true;
	
        document.getElementById("trajectoryButton").textContent ="Hide trails"

    } 

    var event = new Event('FinalizeAndPlayFromOutsideCanvas')
    event.DoClearScreen = args.DoClearScreen
    event.DoXMinMax = args.DoXMinMax
    event.setTinc = false

    displayCanvas.dispatchEvent(event)

    SearchIsOnGoing = false
    var ChoreoExecuteBtn = document.getElementById("ChoreoExecuteBtn")
    ChoreoExecuteBtn.disabled = ""
    var ChoreoDispInitStateBtn = document.getElementById("ChoreoDispInitStateBtn")
    ChoreoDispInitStateBtn.disabled = ""
    var ChoreoSearchNext = document.getElementById("ChoreoSearchNext")
    ChoreoSearchNext.disabled = "disabled"
    AskForNext[0] = 0


    var RotSlider = $("#RotSlider").data("roundSlider")
    RotSlider.enable()

    var Python_State_Div = document.getElementById("Python_State_Div")
    Python_State_Div.innerHTML = "Find solution!"
    Python_State_Div.classList.add('w3-green')
    Python_State_Div.classList.add('w3-hover-pale-green')
    Python_State_Div.classList.remove('w3-orange')
    Python_State_Div.classList.remove('w3-red')

}

function Python_no_sol_found(args) {

	var displayCanvas = document.getElementById("displayCanvas");

    var event = new Event('EnableAnimationFromOutsideCanvas');
    displayCanvas.dispatchEvent(event);

    if (document.getElementById('checkbox_DisplayLoopsDuringSearch').checked) {

        trajectoriesOn = true;
	
        document.getElementById("trajectoryButton").textContent ="Hide trails"

    } 

    SearchIsOnGoing = false
    var ChoreoExecuteBtn = document.getElementById("ChoreoExecuteBtn");
    ChoreoExecuteBtn.disabled = "";
    var ChoreoDispInitStateBtn = document.getElementById("ChoreoDispInitStateBtn");
    ChoreoDispInitStateBtn.disabled = "";
    var ChoreoSearchNext = document.getElementById("ChoreoSearchNext");
    ChoreoSearchNext.disabled = "disabled";
    AskForNext[0] = 0

    var RotSlider = $("#RotSlider").data("roundSlider");
    RotSlider.enable();

    var Python_State_Div = document.getElementById("Python_State_Div");
    Python_State_Div.innerHTML = "Find solution!";
    Python_State_Div.classList.add('w3-green');
    Python_State_Div.classList.add('w3-hover-pale-green');
    Python_State_Div.classList.remove('w3-orange');
    Python_State_Div.classList.remove('w3-red');

}

async function Set_PlotInfo_From_Python(args){

    var txt = await args.JSON_data.text()
    SolName = "Search in progress"
    PlotInfo = JSON.parse(txt)
    UpdateNowPlaying(SearchOnGoing=true)

    var event = new Event('FinalizeSetOrbitFromOutsideCanvas')
    displayCanvas.dispatchEvent(event)

}

async function Plot_Loops_During_Optim_From_Python(args){

	var displayCanvas = document.getElementById("displayCanvas");

    Pos = {"data":args.NPY_data,"shape":args.NPY_shape};

    setPlotWindow(args.Current_PlotWindow);

    var send_event = new Event('DrawAllPathsFromOutsideCanvas');
    displayCanvas.dispatchEvent(send_event);

}

async function Python_Imports_Done(args){

	var displayCanvas = document.getElementById("displayCanvas");

    PythonPrint({txt:"&#10;All python packages imported&#10;"});

    SearchIsOnGoing = false
    var ChoreoExecuteBtn = document.getElementById("ChoreoExecuteBtn");
    ChoreoExecuteBtn.disabled = "";
    var ChoreoDispInitStateBtn = document.getElementById("ChoreoDispInitStateBtn");
    ChoreoDispInitStateBtn.disabled = "";
    var ChoreoSearchNext = document.getElementById("ChoreoSearchNext");
    ChoreoSearchNext.disabled = "disabled";
    AskForNext[0] = 0

    var RotSlider = $("#RotSlider").data("roundSlider");
    RotSlider.enable();
    
    var Python_State_Div = document.getElementById("Python_State_Div");

    Python_State_Div.innerHTML = "Find solution!";
    Python_State_Div.classList.remove('w3-red');
    Python_State_Div.classList.remove('w3-orange');
    Python_State_Div.classList.add('w3-green');
    Python_State_Div.classList.add('w3-hover-pale-green');

    var event = new Event('EnableAnimationFromOutsideCanvas');
    displayCanvas.dispatchEvent(event);

}

pyodide_worker.addEventListener('message', handleMessageFromWorker);

Python_textarea = document.getElementById("Python_textarea")

function OnWindowResize(){

    var CommandBody = document.getElementById("CommandBody");
    var AllLeftTabs = document.getElementsByClassName("LeftTab");
    
    if (window.innerWidth > 1220){
        // Horizontal mode
        CommandBody.style.position = "relative";
        CommandBody.style.marginLeft = "610px";

        if (AllLeftTabs[0].classList.contains("open")) {
            CommandBody.style.bottom = "580px";
        } else {
            CommandBody.style.bottom = "710px";
        }

    } else {
        CommandBody.style.position = "";
        CommandBody.style.marginLeft = "10px";
        CommandBody.style.bottom = "0px";
    }

}

function ClickTopTabBtn(TabId) {
    var i;
    var AllTopTabBtns = document.getElementsByClassName("TopTabBtn");
    var AllMainTabs = document.getElementsByClassName("MainTab");

    for (i = 0; i < AllTopTabBtns.length; i++) { if (AllTopTabBtns[i].classList.contains(TabId)) {AllTopTabBtns[i].classList.add("w3-red");} else {AllTopTabBtns[i].classList.remove("w3-red") ;}}
    for (i = 0; i < AllMainTabs.length  ; i++) { if (AllMainTabs[i].classList.contains(TabId))   {AllMainTabs[i].style.display   = "block";} else {AllMainTabs[i].style.display   = "none"     ;}}
}

function  GeomTopTabBtn(TabId) {
    switch (TabId) {
        case 'Main': {
            ClickTopTabBtn('Main_NowPlaying');
            break;}    
        case 'Geom': {
            ClickTopTabBtn('Geom_Bodies');
            break;}
        case 'Animation': {
            ClickTopTabBtn('Animation_Colors');
            break;}
        case 'Solver': {
            ClickTopTabBtn_Solver_Output();
            break;}
        case 'Misc': {
            ClickTopTabBtn('Misc_Save_Config');
            break;}
    }
}

function ClickLeftTabBtn(TabId) {
    var i;
    var AllLeftTabBtns = document.getElementsByClassName("LeftTabBtn");
    var AllTopTab = document.getElementsByClassName("TopTab");

    for (i = 0; i < AllLeftTabBtns.length; i++) {
         if (AllLeftTabBtns[i].classList.contains(TabId)){
            AllLeftTabBtns[i].classList.add("w3-red");
        } else {
            AllLeftTabBtns[i].classList.remove("w3-red");
        }
    }
    for (i = 0; i < AllTopTab.length     ; i++) {
        if (AllTopTab[i].classList.contains(TabId)){
            AllTopTab[i].style.display     = "block" ;
        } else {
            AllTopTab[i].style.display     = "none"     ;
        }
    }
    GeomTopTabBtn(TabId);
}

var saveJSONData = (function () {
    var a = document.createElement("a");
    document.body.appendChild(a);
    a.style = "display: none";
    return function (data, fileName) {
        var json = JSON.stringify(data,null,2),
            blob = new Blob([json], {type: "octet/stream"}),
            url = window.URL.createObjectURL(blob);
        a.href = url;
        a.download = fileName;
        a.click();
        window.URL.revokeObjectURL(url);
    };
}());

async function SaveConfigFile(UserDir=false){

    var ConfigDict = GatherConfigDict()

    filename = 'choreo_config.json'

    if (UserDir){
            
        ConfigFile = await UserDir.getFileHandle(filename, { create: true })
        const writable = await ConfigFile.createWritable()
        await writable.write(JSON.stringify(ConfigDict,null,2))
        await writable.close()

    } else {
        saveJSONData(ConfigDict, filename)
    }

}

function ChoreoExecuteClick() {

    var ChoreoExecuteBtn = document.getElementById("ChoreoExecuteBtn");

    if (!ChoreoExecuteBtn.disabled) {
        
        SearchIsOnGoing = true
        var ChoreoExecuteBtn = document.getElementById("ChoreoExecuteBtn");
        ChoreoExecuteBtn.disabled = "disabled";
        var ChoreoDispInitStateBtn = document.getElementById("ChoreoDispInitStateBtn");
        ChoreoDispInitStateBtn.disabled = "disabled";        
        var ChoreoSearchNext = document.getElementById("ChoreoSearchNext");
        ChoreoSearchNext.disabled = "";
        AskForNext[0] = 0

        var displayCanvas = document.getElementById("displayCanvas");

        PythonClearPrints();

        var Python_State_Div = document.getElementById("Python_State_Div");

        Python_State_Div.innerHTML = "Next";
        Python_State_Div.classList.add('w3-orange');
        Python_State_Div.classList.remove('w3-green');
        Python_State_Div.classList.remove('w3-hover-pale-green');
        Python_State_Div.classList.remove('w3-red');

        if (document.getElementById('checkbox_DisplayLoopsDuringSearch').checked) {

            var RotSlider = $("#RotSlider").data("roundSlider");
            RotSlider.setValue(0);
            RotSlider.disable();

            var event = new Event('StopAnimationFromOutsideCanvas');
            displayCanvas.dispatchEvent(event);

            var event = new Event('DisableAnimationFromOutsideCanvas');
            displayCanvas.dispatchEvent(event);  

            trajectoriesOn = false;

        }

        var ConfigDict = GatherConfigDict();
        pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{ConfigDict:ConfigDict}});

        ReadyToRun = true

        if (ConfigDict['Geom_Target'] ['LookForTarget']) {

            ReadyToRun = TargetSlow_Loaded

            console.log("TargetSlow_Loaded:",TargetSlow_Loaded)

            if (TargetSlow_Loaded) {

                pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{TargetSlow_PlotInfo:TargetSlow_PlotInfo}});
                pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{TargetSlow_Pos:TargetSlow_Pos}});

                var nfast = TargetSlow_PlotInfo["nloop"]

                for (var i=0; i < nfast; i++) {

                    console.log("TargetFast_LoadedList[i]",i,TargetFast_LoadedList[i])

                    ReadyToRun = ReadyToRun && TargetFast_LoadedList[i]

                }

            }

            if (ReadyToRun) {

                pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{TargetFast_PlotInfoList:TargetFast_PlotInfoList}});
                pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{TargetFast_PosList:TargetFast_PosList}});

            }

        }

        console.log("Ready to run:",ReadyToRun)

        pyodide_worker.postMessage({funname:"ExecutePythonFile",args:"./python_scripts/RunOnce.py"});

    }

}

function Speed_Test_Click() {

    var ConfigDict = GatherConfigDict();
    pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{ConfigDict:ConfigDict}});
    pyodide_worker.postMessage({funname:"ExecutePythonFile",args:"./python_scripts/Speed_Test.py"});

}

function ChoreoSaveInitStateClick() {

    var ConfigDict = GatherConfigDict();

    pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{ConfigDict:ConfigDict}});
    pyodide_worker.postMessage({funname:"ExecutePythonFile",args:"./python_scripts/Load_GUI_params_and_save_init.py"});

}

function GenerateInitStateClick() {

    PythonClearPrints();

    var ConfigDict = GatherConfigDict();

    pyodide_worker.postMessage({funname:"LoadDataInWorker",args:{ConfigDict:ConfigDict}});
    pyodide_worker.postMessage({funname:"ExecutePythonFile",args:"./python_scripts/Save_init_state.py"});

}

function GatherConfigDict() {
    /* Gathers all relevant input in the page and puts it in a dictionary */

    var ConfigDict = {};

    ConfigDict['Geom_Bodies'] = {};

    ConfigDict['Geom_Bodies'] ['MomConsImposed'] = document.getElementById('checkbox_MomCons').checked ;

    table = document.getElementById('table_body_loop');
    var ncols = table.rows[0].cells.length;

    ConfigDict['Geom_Bodies'] ['n_loops'] = ncols - 1;

    ConfigDict['Geom_Bodies'] ['mass'] = [];
    ConfigDict['Geom_Bodies'] ['nbpl'] = [];

    ConfigDict['Geom_Bodies'] ['SymType'] = [];

    for (var icol=1; icol < ncols; icol++) {

        var the_sym = {};
        
        the_sym['n'] =  parseInt(  table.rows[1].cells[icol].children[0].value,10);

        ConfigDict['Geom_Bodies'] ['nbpl'] . push( parseInt(  table.rows[1].cells[icol].children[0].value,10));
        ConfigDict['Geom_Bodies'] ['mass'] . push( parseFloat(table.rows[2].cells[icol].children[0].value)   );
        
        the_sym['name'] = table.rows[3].cells[icol].children[0].value;
        the_sym['k'] = parseInt(table.rows[4].cells[icol].children[0].value,10);
        the_sym['l'] = parseInt(table.rows[5].cells[icol].children[0].value,10);
        the_sym['m'] = parseInt(table.rows[6].cells[icol].children[0].value,10);
        the_sym['p'] = parseInt(table.rows[7].cells[icol].children[0].value,10);
        the_sym['q'] = parseInt(table.rows[8].cells[icol].children[0].value,10);

        ConfigDict['Geom_Bodies'] ['SymType'].push(the_sym);

    }

    ConfigDict['Geom_Target'] = {};
    LookForTarget = document.getElementById('checkbox_Target').checked
    ConfigDict['Geom_Target'] ['LookForTarget'] = LookForTarget
    if (LookForTarget) {

        ConfigDict['Geom_Target'] ['nT_slow'] =  parseInt(document.getElementById('input_nT_slow').value,10)  
        ConfigDict['Geom_Target'] ['nT_fast'] = []

        var nfast = TargetSlow_PlotInfo["nloop"]

        for (var i=0; i < nfast; i++) {

            var id_fast = "input_nT_fast"+i.toString()

            ConfigDict['Geom_Target'] ['nT_fast'].push( parseInt(document.getElementById(id_fast).value,10)  )

        }

    }

    ConfigDict['Geom_Target'] ['Rotate_fast_with_slow'] = document.getElementById('checkbox_RotateFastWithSlow').checked
    ConfigDict['Geom_Target'] ['Randomize_Fast_Init'] = document.getElementById('checkbox_RandomizeFastInit').checked
    ConfigDict['Geom_Target'] ['Optimize_Init'] = document.getElementById('checkbox_OptimizeRelative').checked

    ConfigDict['Geom_Random'] = {};
    ConfigDict['Geom_Random'] ['coeff_ampl_o']    = parseFloat(document.getElementById('input_coeff_ampl_o'   ).value   );
    ConfigDict['Geom_Random'] ['coeff_ampl_min']  = parseFloat(document.getElementById('input_coeff_ampl_min' ).value   );
    ConfigDict['Geom_Random'] ['k_infl']          = parseInt(  document.getElementById('input_k_infl'         ).value,10);
    ConfigDict['Geom_Random'] ['k_max']           = parseInt(  document.getElementById('input_k_max'          ).value,10);

    ConfigDict['Geom_Custom'] = {};

    table = document.getElementById('table_custom_sym');
    var ncols = table.rows[0].cells.length;

    ConfigDict['Geom_Custom'] ['n_custom_sym'] = ncols - 1;
    ConfigDict['Geom_Custom'] ['CustomSyms'] = [];

    for (var icol=1; icol < ncols; icol++) {

        var the_sym = {};
        
        the_sym['LoopTarget']   = parseInt( table.rows[1].cells[icol].children[0].value,10);
        the_sym['LoopSource']   = parseInt( table.rows[2].cells[icol].children[0].value,10);
        the_sym['Reflexion']    =           table.rows[3].cells[icol].children[0].value    ;
        the_sym['RotAngleNum']  = parseInt( table.rows[4].cells[icol].children[0].value,10);
        the_sym['RotAngleDen']  = parseInt( table.rows[5].cells[icol].children[0].value,10);
        the_sym['TimeRev']      =           table.rows[6].cells[icol].children[0].value    ;
        the_sym['TimeShiftNum'] = parseInt( table.rows[7].cells[icol].children[0].value,10);
        the_sym['TimeShiftDen'] = parseInt( table.rows[8].cells[icol].children[0].value,10);

        ConfigDict['Geom_Custom'] ['CustomSyms'].push(the_sym);

    }


    ConfigDict['Animation_Colors'] = {};
    ConfigDict['Animation_Colors'] ["color_method_input"] = document.getElementById("color_method_input").value;

    ConfigDict['Animation_Colors'] ["n_color"] = n_color;
    ConfigDict['Animation_Colors'] ["colorLookup"] = colorLookup;

    ConfigDict['Animation_Size'] = {};
    ConfigDict['Animation_Size'] ['checkbox_Mass_Scale']  = document.getElementById('checkbox_Mass_Scale').checked ;
    ConfigDict['Animation_Size'] ['input_body_radius']  = document.getElementById('input_body_radius').value ;
    ConfigDict['Animation_Size'] ['input_trail_width']  = document.getElementById('input_trail_width').value ;  
    ConfigDict['Animation_Size'] ['checkbox_Trail_vanish']  = document.getElementById('checkbox_Trail_vanish').checked ;
    ConfigDict['Animation_Size'] ['input_trail_vanish_length']  = document.getElementById('input_trail_vanish_length').value;

    ConfigDict['Animation_Framerate'] = {};
    ConfigDict['Animation_Framerate'] ['checkbox_Limit_FPS']  = document.getElementById('checkbox_Limit_FPS').checked  ;
    ConfigDict['Animation_Framerate'] ['input_Limit_FPS']  = parseInt(document.getElementById('input_Limit_FPS').value,10)        ;

    ConfigDict['Animation_Search'] = {};
    ConfigDict['Animation_Search'] ['DisplayLoopsDuringSearch']  = document.getElementById('checkbox_DisplayLoopsDuringSearch').checked  ;
    ConfigDict['Animation_Search'] ['DisplayBodiesDuringSearch']  = document.getElementById('checkbox_DisplayBodiesDuringSearch').checked  ;
    ConfigDict['Animation_Search'] ['DisplayLoopOnGalleryLoad']  = document.getElementById('checkbox_DisplayLoopOnGalleryLoad').checked  ;

    ConfigDict['Solver_Discr'] = {};
    ConfigDict['Solver_Discr'] ['Use_exact_Jacobian']  = document.getElementById('checkbox_exactJ').checked            ;
    ConfigDict['Solver_Discr'] ['ncoeff_init']         = parseInt(document.getElementById('input_ncoeff_init').value,10)        ;
    ConfigDict['Solver_Discr'] ['n_reconverge_it_max'] = parseInt(document.getElementById('input_n_reconverge_it_max').value,10);

    ConfigDict['Solver_Optim'] = {};

    ConfigDict['Solver_Optim'] ['krylov_method']  = document.getElementById('krylov_method').value;
    ConfigDict['Solver_Optim'] ['line_search']    = document.getElementById('linesearch_method').value;
    ConfigDict['Solver_Optim'] ['line_search_smin']    = parseFloat(document.getElementById('linesearch_smin').value);

    ConfigDict['Solver_Optim'] ['Newt_err_norm_max'] = parseFloat(document.getElementById('input_Newt_err_norm_max').value);
    ConfigDict['Solver_Optim'] ['optim_verbose_lvl'] = document.getElementById('optim_verbose_lvl').value;

    ConfigDict['Solver_Loop'] = {};

    table = document.getElementById('table_cvg_loop');
    var ncols = table.rows[0].cells.length;

    ConfigDict['Solver_Loop'] ['n_optim_param'] = ncols - 1;
    ConfigDict['Solver_Loop'] ['gradtol_list'] = [];
    ConfigDict['Solver_Loop'] ['inner_maxiter_list'] = [];
    ConfigDict['Solver_Loop'] ['maxiter_list'] = [];
    ConfigDict['Solver_Loop'] ['outer_k_list'] = [];
    ConfigDict['Solver_Loop'] ['store_outer_Av_list'] = [];

    for (var icol=1; icol < ncols; icol++) {

        ConfigDict['Solver_Loop'] ['gradtol_list']       . push( parseFloat(table.rows[1].cells[icol].children[0].value   ));
        ConfigDict['Solver_Loop'] ['maxiter_list']       . push( parseInt(  table.rows[2].cells[icol].children[0].value,10));
        ConfigDict['Solver_Loop'] ['inner_maxiter_list'] . push( parseInt(  table.rows[3].cells[icol].children[0].value,10));
        ConfigDict['Solver_Loop'] ['outer_k_list']       . push( parseInt(  table.rows[4].cells[icol].children[0].value,10));
        
        ConfigDict['Solver_Loop'] ['store_outer_Av_list']. push( table.rows[5].cells[icol].children[0].value == "True");

    }

    ConfigDict['Solver_Checks'] = {};

    ConfigDict['Solver_Checks'] ['Look_for_duplicates'] = document.getElementById('checkbox_duplicates').checked;
    ConfigDict['Solver_Checks'] ['duplicate_eps']       = parseFloat(document.getElementById('input_duplicate_eps').value);
    ConfigDict['Solver_Checks'] ['Check_Escape']        = document.getElementById('checkbox_escape').checked;

    return ConfigDict;
}

function ClickLoadConfigFile(files) {
    files = [...files];
    files.forEach(LoadConfigFile);
}

function DropConfigFile(e) {
    var dt = e.dataTransfer;
    var files = dt.files;
    ClickLoadConfigFile(files);
}

function PreventDefaultDragOver(event) {
    event.preventDefault();
}

async function LoadConfigFile(the_file) {

    var txt = await the_file.text();

    var ConfigDict = JSON.parse(txt);

    LoadConfigDict(ConfigDict);

}

function LoadConfigDict(ConfigDict) {

    var displayCanvas = document.getElementById("displayCanvas")

    var table = document.getElementById('table_body_loop')
    var ncols = table.rows[0].cells.length

    for (var icol=ncols-1; icol > 0; icol-- ) {
        deleteColumn('table_body_loop',icol)
    };

    document.getElementById('checkbox_MomCons').checked  = ConfigDict['Geom_Bodies'] ['MomConsImposed'] 

    var n_loops = ConfigDict['Geom_Bodies'] ['n_loops']

    for (var il=0; il < n_loops; il++) {

        ClickAddBodyLoop()

        var icol = il+1

        table.rows[1].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['nbpl'] [il]
        table.rows[2].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['mass'] [il] . toString()
        table.rows[3].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['SymType'] [il] ['name']
        table.rows[4].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['SymType'] [il] ['k']
        table.rows[5].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['SymType'] [il] ['l']
        table.rows[6].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['SymType'] [il] ['m']
        table.rows[7].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['SymType'] [il] ['p']
        table.rows[8].cells[icol].children[0].value = ConfigDict['Geom_Bodies'] ['SymType'] [il] ['q']
        
    }

    RedistributeClicksTableBodyLoop('table_body_loop',1,RedistributeBodyCount)


    document.getElementById('checkbox_Target'            ).checked = ConfigDict['Geom_Target'] ['LookForTarget']
    document.getElementById('checkbox_RotateFastWithSlow').checked = ConfigDict['Geom_Target'] ['Rotate_fast_with_slow']
    document.getElementById('checkbox_RandomizeFastInit' ).checked = ConfigDict['Geom_Target'] ['Randomize_Fast_Init']
    document.getElementById('checkbox_OptimizeRelative'  ).checked = ConfigDict['Geom_Target'] ['Optimize_Init']

    document.getElementById('input_coeff_ampl_o'   ).value = ConfigDict['Geom_Random'] ['coeff_ampl_o']   
    document.getElementById('input_coeff_ampl_min' ).value = ConfigDict['Geom_Random'] ['coeff_ampl_min'] 
    document.getElementById('input_k_infl'         ).value = ConfigDict['Geom_Random'] ['k_infl']         
    document.getElementById('input_k_max'          ).value = ConfigDict['Geom_Random'] ['k_max']          

    var table = document.getElementById('table_custom_sym')
    var ncols = table.rows[0].cells.length

    for (var icol=ncols-1; icol > 0; icol-- ) {
        deleteColumn('table_custom_sym',icol)
    };

    var nsym = ConfigDict['Geom_Custom'] ['n_custom_sym']

    for (var isym=0; isym < nsym; isym++) {

        ClickAddCustomSym()

        var icol = isym+1

        table.rows[1].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['LoopTarget']   
        table.rows[2].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['LoopSource']   
        table.rows[3].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['Reflexion']    
        table.rows[4].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['RotAngleNum']  
        table.rows[5].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['RotAngleDen']  
        table.rows[6].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['TimeRev']      
        table.rows[7].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['TimeShiftNum'] 
        table.rows[8].cells[icol].children[0].value = ConfigDict['Geom_Custom'] ['CustomSyms'] [isym] ['TimeShiftDen'] 

    }

    RedistributeClicksTableBodyLoop('table_custom_sym',0)

    document.getElementById("color_method_input").value = ConfigDict['Animation_Colors'] ["color_method_input"]
    
    var n_color_old = n_color
    for (var i_color=0;i_color<n_color_old; i_color++) {
        RemoveColor()
    }

    var n_color_new = ConfigDict['Animation_Colors'] ["n_color"]
    for (var i_color=0;i_color<n_color_new; i_color++) {
        AddColor(ConfigDict['Animation_Colors'] ["colorLookup"][i_color] )
    }

    document.getElementById('checkbox_Mass_Scale').checked  = ConfigDict['Animation_Size'] ['checkbox_Mass_Scale']
    DoScaleSizeWithMass = ConfigDict['Animation_Size'] ['checkbox_Mass_Scale']
    document.getElementById('input_body_radius').value = ConfigDict['Animation_Size'] ['input_body_radius'] 
    SlideBodyRadius()
    document.getElementById('input_trail_width').value = ConfigDict['Animation_Size'] ['input_trail_width']
    SlideTrailWidth()
    document.getElementById('checkbox_Trail_vanish').checked  = ConfigDict['Animation_Size'] ['checkbox_Trail_vanish']
    DoTrailVanish = ConfigDict['Animation_Size'] ['checkbox_Trail_vanish']
    document.getElementById('input_trail_vanish_length').value = ConfigDict['Animation_Size'] ['input_trail_vanish_length']
    // SlideTrailTime()
    
    ExportColors()
    var send_event = new Event('ChangeColorsFromOutsideCanvas')
    displayCanvas.dispatchEvent(send_event)

    document.getElementById('checkbox_Limit_FPS').checked = ConfigDict['Animation_Framerate']['checkbox_Limit_FPS'] 
    document.getElementById('input_Limit_FPS').value = ConfigDict['Animation_Framerate'] ['input_Limit_FPS'] 

    document.getElementById('checkbox_DisplayLoopsDuringSearch').checked = ConfigDict['Animation_Search'] ['DisplayLoopsDuringSearch']
    if (document.getElementById('checkbox_DisplayLoopsDuringSearch').checked) {
        document.getElementById('checkbox_DisplayBodiesDuringSearch').disabled = ""
    } else {
        document.getElementById('checkbox_DisplayBodiesDuringSearch').disabled = "disabled"
    }

    document.getElementById('checkbox_DisplayBodiesDuringSearch').checked = ConfigDict['Animation_Search'] ['DisplayBodiesDuringSearch']
    document.getElementById('checkbox_DisplayLoopOnGalleryLoad').checked  = ConfigDict['Animation_Search'] ['DisplayLoopOnGalleryLoad']

    document.getElementById('checkbox_exactJ').checked         = ConfigDict['Solver_Discr'] ['Use_exact_Jacobian']  
    document.getElementById('input_ncoeff_init').value         = ConfigDict['Solver_Discr'] ['ncoeff_init']         
    document.getElementById('input_n_reconverge_it_max').value = ConfigDict['Solver_Discr'] ['n_reconverge_it_max'] 

    SlideNReconvergeItMax();

    document.getElementById('krylov_method').value           = ConfigDict['Solver_Optim'] ['krylov_method']     
    document.getElementById('linesearch_method').value       = ConfigDict['Solver_Optim'] ['line_search']       
    document.getElementById('linesearch_smin').value         = ConfigDict['Solver_Optim'] ['line_search_smin']  
    document.getElementById('input_Newt_err_norm_max').value = ConfigDict['Solver_Optim'] ['Newt_err_norm_max'] 
    document.getElementById('optim_verbose_lvl').value       = ConfigDict['Solver_Optim'] ['optim_verbose_lvl'] 

    var table = document.getElementById('table_cvg_loop')
    var ncols = table.rows[0].cells.length

    for (var icol=ncols-1; icol > 0; icol-- ) {
        deleteColumn('table_cvg_loop',icol)
    };

    var n_loops = ConfigDict['Solver_Loop'] ['n_optim_param']

    for (var il=0; il < n_loops; il++) {

        ClickAddColLoopKrylov()

        var icol = il+1

        table.rows[1].cells[icol].children[0].value = ConfigDict['Solver_Loop'] ['gradtol_list'] [il] 
        table.rows[2].cells[icol].children[0].value = ConfigDict['Solver_Loop'] ['maxiter_list'] [il] 
        table.rows[3].cells[icol].children[0].value = ConfigDict['Solver_Loop'] ['inner_maxiter_list'] [il] 
        table.rows[4].cells[icol].children[0].value = ConfigDict['Solver_Loop'] ['outer_k_list'] [il] 

        if (ConfigDict['Solver_Loop'] ['store_outer_Av_list'] [il]) {
        table.rows[5].cells[icol].children[0].value = "True" 
        } else {
        table.rows[5].cells[icol].children[0].value = "False" 
        }

    }

    RedistributeClicksTableBodyLoop('table_cvg_loop',1)

    document.getElementById('checkbox_duplicates').checked = ConfigDict['Solver_Checks'] ['Look_for_duplicates'] 
    document.getElementById('input_duplicate_eps').value   = ConfigDict['Solver_Checks'] ['duplicate_eps']       
    document.getElementById('checkbox_escape').checked     = ConfigDict['Solver_Checks'] ['Check_Escape']        

}

function SlideNReconvergeItMax() {
    var slider = document.getElementById("input_n_reconverge_it_max");
    var output = document.getElementById("display_n_reconverge_it_max");
    output.innerHTML = slider.value;
}

function OpenCloseLeftTab() {
    var i;
    var AllLeftTabs = document.getElementsByClassName("LeftTab");
    if (AllLeftTabs[0].classList.contains("open")) {CloseLeftTab();} else {OpenLeftTab();}
}

canvas_items_list= ["canvasContainer","displayCanvas","particleLayerCanvas"]

function CloseLeftTab() {
    var i;
    var AllLeftTabs        = document.getElementsByClassName("LeftTab");
    var MarginLeftTop      = document.getElementById("MarginLeftTop");
    var AllLeftTabBtns     = document.getElementsByClassName("LeftTabBtn");
    var AnimationBlock     = document.getElementById("AnimationBlock");
    var AllTopTabs         = document.getElementsByClassName("TopTab");
    var CommandBody        = document.getElementById("CommandBody");
    for (i = 0; i < AllLeftTabs.length     ; i++) {
        AllLeftTabs[i].classList.remove("open");
        AllLeftTabs[i].classList.add("closed");
        AllLeftTabs[i].style.width     = "43px"     ;
    }
    MarginLeftTop.style.marginLeft      = "43px"     ;
    for (i = 0; i < AllLeftTabBtns.length; i++) {
        AllLeftTabBtns[i].style.display     = "none";
    }
    AnimationBlock.style.marginLeft      = "0px"     ;
    for (i = 0; i < AllTopTabs.length; i++) {
        AllTopTabs[i].style.width     = "567px";
    }
    for (i = 0; i < canvas_items_list.length; i++) {
        var canvas_item = document.getElementById(canvas_items_list[i]);
        canvas_item.style.width     = "610px"     ;
        canvas_item.style.height     = "610px"     ;
    }
    if (window.innerWidth > 1220){
        CommandBody.style.bottom = "710px";
    } 
    
}

function OpenLeftTab() {
    var i;
    var AllLeftTabs        = document.getElementsByClassName("LeftTab");
    var MarginLeftTop      = document.getElementById("MarginLeftTop");
    var AllLeftTabBtns     = document.getElementsByClassName("LeftTabBtn");
    var AnimationBlock     = document.getElementById("AnimationBlock");
    var AllTopTabs         = document.getElementsByClassName("TopTab");
    var CommandBody        = document.getElementById("CommandBody");
    for (i = 0; i < AllLeftTabs.length     ; i++) {
        AllLeftTabs[i].classList.add("open");
        AllLeftTabs[i].classList.remove("closed");
        AllLeftTabs[i].style.width     = "130px"     ;
    }
    MarginLeftTop.style.marginLeft      = "130px"     ;
    for (i = 0; i < AllLeftTabBtns.length; i++) {
        AllLeftTabBtns[i].style.display     = "";
    }
    AnimationBlock.style.marginLeft      = "130px"     ;
    for (i = 0; i < AllTopTabs.length; i++) {
        AllTopTabs[i].style.width     = "480px";
    }
    for (i = 0; i < canvas_items_list.length; i++) {
        var canvas_item = document.getElementById(canvas_items_list[i]);
        canvas_item.style.width     = "480px"     ;
        canvas_item.style.height     = "480px"     ;
    }
    if (window.innerWidth > 1220){
        CommandBody.style.bottom = "580px";
    } 
}    

function deleteColumn(tableID, colnum) {
    var table = document.getElementById(tableID);
    var i;

    if (colnum < table.rows[0].cells.length) {
        for (i = 0; i < table.rows.length; i++) {
            table.rows[i].deleteCell(colnum);
        }
    }
}

function deleteLastColumn(tableId) {
    var lastCol = document.getElementById(tableId).rows[0].cells.length - 1;
    deleteColumn(tableId, lastCol);
}

function RedistributeClicksTableBodyLoop(tableid,mincol,fun_exec_end=function(){}){
    var table = document.getElementById(tableid)

    for (var icol=1; icol < table.rows[0].cells.length; icol++) {
        
        var div = table.rows[0].cells[icol].children[0]
        div.button_number = icol
        
        div.onclick = function () {
            if (table.rows[0].cells.length > (mincol+1)) {
                deleteColumn(tableid, this.button_number)
                RedistributeClicksTableBodyLoop(tableid,mincol,fun_exec_end)
            }
        }

    }

    fun_exec_end()

}

function RedistributeBodyCount() {
    var table = document.getElementById('table_body_loop')

    var irow_n_body = 1
    var irow_body_range = 9
    var ibody_low = 0
    var ibody_high = 0                                   

    for (var icol=1; icol < table.rows[0].cells.length; icol++) {
        
        var nbody = parseInt(table.rows[irow_n_body].cells[icol].children[0].value,10);
        ibody_high = ibody_high + nbody -1 ;

        table.rows[irow_body_range].cells[icol].children[0].innerHTML = ibody_low.toString() + " - " + ibody_high.toString();

        ibody_low = ibody_high+1;
        ibody_high = ibody_low;
    }
}

function ClickAddColLoopKrylov() {
    var table = document.getElementById('table_cvg_loop')
    var newcell
    var div,input
    var irow, ival
    var icol = table.rows[0].cells.length

    var input_dict = [
        {
        "elem_class":"input", 
        "type":"text", 
        "value":["1e-1"   ,"1e-3"   ,"1e-5"   ,"1e-7"   ,"1e-9"   ,"1e-11"  ,"1e-13"  ,"1e-15"  ],
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":["100"    ,"1000"   ,"1000"   ,"1000"   ,"500"     ,"500"    ,"300"   ,"100"    ],
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":["30"     ,"30"     ,"50"     ,"60"     ,"70"     ,"80"     ,"100"    ,"100"    ],
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":["5"      ,"5"      ,"5"      ,"5"      ,"5"      ,"7"      ,"7"      ,"7"      ],
        },
        {
        "elem_class":"select", 
        "class":"w3-select",  
        "innerHTML":[
            "<option value='True' selected>True</option><option value='False'>False</option>",
            "<option value='True' selected>True</option><option value='False'>False</option>",
            "<option value='True' selected>True</option><option value='False'>False</option>",
            "<option value='True' selected>True</option><option value='False'>False</option>",
            "<option value='True' selected>True</option><option value='False'>False</option>",
            "<option value='True'>True</option><option value='False' selected>False</option>",
            "<option value='True'>True</option><option value='False' selected>False</option>",
            "<option value='True'>True</option><option value='False' selected>False</option>",
        ]
        },
    ]

    n_fields = input_dict.length
    n_default_values = input_dict[0]['value'].length

    irow = 0
    newcell = table.rows[irow].insertCell(icol)
    newcell.style.borderLeftStyle = 'hidden'
    newcell.style.fontSize = '16px'
    newcell.style.width = '65px'
    newcell.style.textAlign = 'center'

    div = document.createElement('button')
    div.classList.add("w3-button","w3-light-grey","w3-hover-pale-red")
    div.style.textAlign = "center"
    div.style.fontSize ="16px"
    div.style.fontWeight ="bold"
    div.innerHTML = "-"

    newcell.appendChild(div)

    for (ival = 0; ival < n_fields; ival++) {
        irow = ival + 1
        newcell = table.rows[irow].insertCell(icol)
        newcell.style.width = '65px'
        newcell.style.textAlign = 'center'
        var idx = Math.min(icol-1,n_default_values-1)
        input = document.createElement(input_dict[ival]["elem_class"])
        for (var [key, val] of Object.entries(input_dict[ival])){
            if (key != "elem_class"){
                if (Array.isArray(val)){
                    input[key] = val[idx]
                }
                else{
                    input[key] = val
                }
            }
            input.style = "width: 53px; text-align: center;"
        }
        newcell.appendChild(input)
    }

    RedistributeClicksTableBodyLoop('table_cvg_loop',1)

}

function ClickAddBodyLoop() {
    var table = document.getElementById('table_body_loop')
    var newcell
    var div,input
    var irow, ival, jcol
    var icol = table.rows[0].cells.length

    var input_dict = [
        {
            "elem_class":"input", 
            "type":"number", 
            "value":"3",
            "min":"1",
            "oninput":"RedistributeBodyCount",
        },
        {
            "elem_class":"input", 
            "type":"text", 
            "value":"1.",
        },
        {
            "elem_class":"select", 
            "class":"w3-select",  
            "innerHTML":"<option value='C' selected>C</option><option value='D'>D</option><option value='Cp'>Cp</option><option value='Dp'>Dp</option>",
        },
        {
            "elem_class":"input", 
            "type":"number", 
            "min":"1",
            "value":"1",
        },
        {
            "elem_class":"input", 
            "type":"number", 
            "min":"0",
            "value":"1",
        },
        {
            "elem_class":"input", 
            "type":"number", 
            "min":"0",
            "value":"1",
        },
        {
            "elem_class":"input", 
            "type":"number", 
            "min":"0",
            "value":"1",
        },
        {
            "elem_class":"input", 
            "type":"number", 
            "min":"1",
            "value":"1",
        },
        {
            "elem_class":"text",
            "max-width":"60px",
            "innerHTML":"",
        },
    ]

    n_fields = input_dict.length

    irow = 0
    newcell = table.rows[irow].insertCell(icol)
    newcell.style.borderLeftStyle = 'hidden'
    newcell.style.fontSize = '16px'
    newcell.style.width = '60px'
    newcell.style.textAlign = 'center'

    div = document.createElement('button')
    div.classList.add("w3-button","w3-light-grey","w3-hover-pale-red","TargetToggle")
    div.style.textAlign = "center"
    div.style.fontSize ="16px"
    div.style.fontWeight ="bold"
    div.innerHTML = "-"

    newcell.appendChild(div);

    for (ival = 0; ival < n_fields; ival++) {
        irow = ival + 1
        newcell = table.rows[irow].insertCell(icol)
        newcell.style.width = '60px'
        newcell.style.textAlign = 'center';  
        input = document.createElement(input_dict[ival]["elem_class"])
        input.classList.add("TargetToggle")
        for (var [key, val] of Object.entries(input_dict[ival])){
            if (key != "elem_class"){
                input[key] = val
            }
            if (key == "oninput"){
                input[key] = window[val]
            }
            input.style = "width: 45px; text-align: center;"
        }
        newcell.appendChild(input)
    }

    RedistributeClicksTableBodyLoop('table_body_loop',1,RedistributeBodyCount)

}

function ClickAddCustomSym() {
    var table = document.getElementById('table_custom_sym');
    var newcell;
    var div,input;
    var irow, ival, jcol;
    var icol = table.rows[0].cells.length;

    var input_dict = [
        {
        "elem_class":"input", 
        "type":"number", 
        "value":"1",
        "min":"0",
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":"0",
        "min":"0",
        },
        {
        "elem_class":"select", 
        "class":"w3-select",  
        "innerHTML":"<option value='True'>True</option><option value='False' selected>False</option>",
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":"1",
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":"1",
        },
        {
        "elem_class":"select", 
        "class":"w3-select",  
        "innerHTML":"<option value='True'>True</option><option value='False' selected>False</option>",
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":"1",
        },
        {
        "elem_class":"input", 
        "type":"number", 
        "value":"1",
        }
    ];

    n_fields = input_dict.length;

    irow = 0;
    newcell = table.rows[irow].insertCell(icol);
    newcell.style.borderLeftStyle = 'hidden';
    newcell.style.fontSize = '16px';
    newcell.style.width = '65px';
    newcell.style.textAlign = 'center';

    div = document.createElement('button'); 
    div.classList.add("w3-button");
    div.classList.add("w3-light-grey");
    div.classList.add("w3-hover-pale-red");
    div.style.textAlign = "center";
    div.style.fontSize ="16px";
    div.style.fontWeight ="bold";
    div.innerHTML = "-";

    newcell.appendChild(div);

    for (ival = 0; ival < n_fields; ival++) {
        irow = ival + 1;
        newcell = table.rows[irow].insertCell(icol);
        newcell.style.width = '65px';
        newcell.style.textAlign = 'center';   
        input = document.createElement(input_dict[ival]["elem_class"]);
        for (var [key, val] of Object.entries(input_dict[ival])){
        if (key != "elem_class"){
            input[key] = val;
        }
        input.style = "width: 53px; text-align: center;"
        }
        newcell.appendChild(input);
    }

    RedistributeClicksTableBodyLoop('table_custom_sym',0);

}

var ncol_per_row_colortable = 4;

function ClickRemoveColor() {

    if (n_color > 1) {

        var displayCanvas = document.getElementById("displayCanvas");

        RemoveColor();
        ExportColors();

        var send_event = new Event('ChangeColorsFromOutsideCanvas');
        displayCanvas.dispatchEvent(send_event);

    }

}
  
function RemoveColor() {

    var table = document.getElementById('table_pick_color');

    var nrow_cur = table.rows.length;

    var icol = ((n_color -1) % ncol_per_row_colortable) ;
    var irow = Math.floor((n_color-1) / ncol_per_row_colortable) +1;

    table.rows[irow].deleteCell(icol);

    if (icol == 0) {
        table.deleteRow(irow);
    }

    n_color -= 1;

}
  
function ClickAddColor() {

    var displayCanvas = document.getElementById("displayCanvas");

    AddColor();
    ExportColors();

    var send_event = new Event('ChangeColorsFromOutsideCanvas');
    displayCanvas.dispatchEvent(send_event);
}

function AddColor(the_color) {
    var table = document.getElementById('table_pick_color');
    var newcell;
    var div,input;
    var ival;
    var color;

    var nrow_cur = table.rows.length;

    n_color += 1;

    var icol = ((n_color -1) % ncol_per_row_colortable) ;
    var irow = Math.floor((n_color-1) / ncol_per_row_colortable) +1;

    if (irow >= nrow_cur) {
        table.insertRow();
        table.rows[irow].style.borderStyle = 'hidden';
    }

    newcell = table.rows[irow].insertCell(icol);
    newcell.style.borderStyle = 'hidden';
    newcell.style.fontSize = '16px';
    newcell.style.width = '65px';
    newcell.style.textAlign = 'center';

    /* Color number text */
    div = document.createElement('div'); 
    div.style.fontSize ="16px";
    div.style.display ="inline-block";
    div.style.width ="35px";
    div.innerHTML = (n_color -1).toString()+": ";
    newcell.appendChild(div);

    /* Color input  */
    div = document.createElement('input'); 
    div.type = "color";
    div.style.display ="inline-block";
    div.style.width ="80px";
    div.classList.add("particle_color_picker");

    if (the_color !== undefined) {

        color = the_color

    } else {

        if (n_color <= colorLookup_init.length) {
            color = colorLookup_init[n_color-1]
        } else {
            color = defaultParticleColor;
        }
    
    }

    div.value = color;
    div.targetid = n_color-1;
    div.addEventListener("input", ChangeColor_Handler, true);
    newcell.appendChild(div);

}

function ExportColors() {

    colorLookup = new Array(n_color);

    var AllColorPickers = document.getElementsByClassName("particle_color_picker");

    for (var ipick=0; ipick < AllColorPickers.length ; ipick++) {

        var targetid = AllColorPickers[ipick].targetid;
        var color = AllColorPickers[ipick].value;    
        
        colorLookup[targetid] = color;

    }

}

color_method_input = document.getElementById("color_method_input");
color_method_input.addEventListener("input", color_method_input_Handler, true);

function color_method_input_Handler(event) {

    var displayCanvas = document.getElementById("displayCanvas");

    var send_event = new Event('ChangeColorsFromOutsideCanvas');
    displayCanvas.dispatchEvent(send_event);

}

function ChangeColor_Handler(event) {

    var displayCanvas = document.getElementById("displayCanvas");

    var targetid = event.path[0].targetid;
    var color = event.path[0].value;

    colorLookup[targetid] = color;
    
    var send_event = new Event('ChangeColorsFromOutsideCanvas');
    displayCanvas.dispatchEvent(send_event);

}

function ClickTopTabBtn_Animation_Framerate() {
    UpdateFPSDisplay();
    ClickTopTabBtn('Animation_Framerate');
}

function ClickTopTabBtn_Solver_Output() {

    ClickTopTabBtn('Solver_Output')

    Python_textarea.scrollTop = Python_textarea.scrollHeight

}

function UpdateFPSDisplay() {

    var the_time = performance.now();

    if (the_time > (Last_UpdateFPSDisplay + 1000/UpdateFPSDisplay_freq)) {

        Last_UpdateFPSDisplay = the_time;

        var txt_disp = document.getElementById('Estimate_FPS_txt');
        txt_disp.innerHTML=Math.ceil(FPS_estimation).toString();
        
    }

}

function KillAndReloadWorker() {
    
    pyodide_worker.terminate();

    SearchIsOnGoing = false
    var ChoreoExecuteBtn = document.getElementById("ChoreoExecuteBtn");
    ChoreoExecuteBtn.disabled = "disabled";
    var ChoreoDispInitStateBtn = document.getElementById("ChoreoDispInitStateBtn");
    ChoreoDispInitStateBtn.disabled = "disabled";
    var ChoreoSearchNext = document.getElementById("ChoreoSearchNext");
    ChoreoSearchNext.disabled = "disabled";
    AskForNext[0] = 0

    var Python_State_Div = document.getElementById("Python_State_Div");

    PythonClearPrints();
    PythonPrint({txt:"Python Killed. Reloading ...&#10;"});

    Python_State_Div.innerHTML = "Killed";
    Python_State_Div.classList.add('w3-red');
    Python_State_Div.classList.remove('w3-orange');
    Python_State_Div.classList.remove('w3-green');
    Python_State_Div.classList.remove('w3-hover-pale-green');

    pyodide_worker = new Worker("./Pyodide_worker.js");
    pyodide_worker.addEventListener('message', handleMessageFromWorker);
    pyodide_worker.postMessage({funname:"ExecutePythonFile",args:"./python_scripts/Python_imports.py"});
    pyodide_worker.postMessage({funname: "setAskForNextBuffer",args:AskForNextBuffer });

}

function ChoreoSearchNextClick(){

    AskForNext[0] = 1

}

var checkbox_Limit_FPS = document.getElementById('checkbox_Limit_FPS');
checkbox_Limit_FPS.addEventListener("change", checkbox_Limit_FPS_Handler, true);

var checkbox_Target = document.getElementById('checkbox_Target');
checkbox_Target.addEventListener("change", checkbox_EnableTargets_Handler, true);

var input_Limit_FPS = document.getElementById("input_Limit_FPS");
input_Limit_FPS.addEventListener("change", input_Limit_FPS_Handler, true);

var checkbox_Mass_Scale = document.getElementById('checkbox_Mass_Scale');
checkbox_Mass_Scale.addEventListener("change", checkbox_Mass_Scale_Handler, true);

var input_body_radius = document.getElementById("input_body_radius");
input_body_radius.addEventListener("input" , SlideBodyRadius, true);
input_body_radius.value = base_particle_size ;
input_body_radius.min   = min_base_particle_size ;
input_body_radius.max   = max_base_particle_size ;
input_body_radius.step  = 0.05 ;

var input_trail_width = document.getElementById("input_trail_width");
input_trail_width.addEventListener("input" , SlideTrailWidth, true);
input_trail_width.value = base_trailWidth ;
input_trail_width.min   = min_base_trailWidth ;
input_trail_width.max   = max_base_trailWidth ;
input_trail_width.step  = 0.05 ;

var checkbox_Trail_vanish = document.getElementById('checkbox_Trail_vanish');
checkbox_Trail_vanish.addEventListener("change", checkbox_Trail_vanish_Handler, true);

var input_trail_vanish_length = document.getElementById("input_trail_vanish_length");
input_trail_vanish_length.addEventListener("input" , SlideTrailTime, true);
input_trail_vanish_length.value = base_trail_vanish_length ;
input_trail_vanish_length.min   = min_base_trail_vanish_length ;
input_trail_vanish_length.max   = max_base_trail_vanish_length ;
input_trail_vanish_length.step  = 0.05 ;

function SlideBodyRadius(event) {
    base_particle_size = input_body_radius.value;
}

function SlideTrailWidth(event) {
    
    base_trailWidth = input_trail_width.value;
}

function SlideTrailTime(event) {
    
    trail_vanish_length = input_trail_vanish_length.value;
    

    var dx = xMax - xMin
    var dy = yMax - yMin
    var distance_ref = Math.sqrt(dx*dx + dy*dy)
    var distance_rel = Max_PathLength / distance_ref

    FadeInvFrequency = (trail_vanish_length_mul * trail_vanish_length) / distance_rel ;

}

function checkbox_Mass_Scale_Handler(event) {

    var displayCanvas = document.getElementById("displayCanvas")

    DoScaleSizeWithMass = event.currentTarget.checked
    
    var event = new Event('RemakeParticlesFromOutsideCanvas')
    displayCanvas.dispatchEvent(event)

}

function checkbox_Trail_vanish_Handler(event) {

    DoTrailVanish = event.currentTarget.checked;
    LastFadeTime = 0
    
}

function input_Limit_FPS_Handler() {

    Elapsed_Time_During_Animation = 0
    n_valid_dt_animation = 0

    FPS_limit = input_Limit_FPS.value

    anim_schedule_time = performance.now()

}

function checkbox_Limit_FPS_Handler(event) {
    
    if (event.currentTarget.checked) {
        input_Limit_FPS.disabled = ""
        Do_Limit_FPS = true
    } else {
        input_Limit_FPS.disabled = "disabled"
        Do_Limit_FPS = false
    }

    input_Limit_FPS_Handler()

}

var checkbox_Cookie = document.getElementById('checkbox_Cookie');
checkbox_Cookie.addEventListener("change", checkbox_Cookie_Handler, true);

var Save_Cookie_Btn = document.getElementById('Save_Cookie_Btn');
var Load_Cookie_Btn = document.getElementById('Load_Cookie_Btn');

var Cookie_Message = document.getElementById('Cookie_Message');

var cookie_name = "choreo_GUI"

function checkbox_Cookie_Handler(event) {

    if (event.currentTarget.checked) {

        Save_Cookie_Btn.disabled = "";
        Load_Cookie_Btn.disabled = "";

    } else {

        Save_Cookie_Btn.disabled = "disabled";
        Load_Cookie_Btn.disabled = "disabled";

        txt = LoadCookie(cookie_name);

        if (txt == "") {

            IssueCookieMessage("There is no cookie on your device.");

        } else {

            DeleteCookie(cookie_name);
            IssueCookieMessage("Cookie deleted !");

        }
    }

}

async function IssueCookieMessage(message,keep_up) {

    Cookie_Message.innerHTML = message;

    if (keep_up !== undefined) {
    } else {
        setTimeout(function(){IssueCookieMessage("",true)}, 3000);
    }

}

function SaveCookieClick(event) {

    var ConfigDict = GatherConfigDict();

    var cookie_value = JSON.stringify(ConfigDict,null,0)
    var cookie_lifetime = 365;

    SaveCookie(cookie_name, cookie_value, cookie_lifetime);

    IssueCookieMessage("Cookie saved !");
}

function LoadCookieClick(event) {

    success = DealWithCookie();
    if (success) {
        IssueCookieMessage("Cookie loaded !");
    } else {
        IssueCookieMessage("Cookie not found !");
    }

}

function SaveCookie(name, value, time_expires_days) {
    var d = new Date();
    d.setTime(d.getTime() + (time_expires_days*24*60*60*1000));
    var expires = "expires="+ d.toUTCString();
    document.cookie = name + "=" + value + ";" + expires + ";path=/";
}

function DealWithCookie() {

    var txt = LoadCookie(cookie_name);

    if (txt == "") {
    
        return false;

    } else {

        checkbox_Cookie.checked = true;
        Save_Cookie_Btn.disabled = "";
        Load_Cookie_Btn.disabled = "";    

        var ConfigDict = JSON.parse(txt);

        LoadConfigDict(ConfigDict);

        return true;

    }

}

function LoadCookie(name) {
    let head = name + "="
    let decodedCookie = decodeURIComponent(document.cookie)
    let ca = decodedCookie.split(';')
    for(let i = 0; i <ca.length; i++) {
        let c = ca[i]
        while (c.charAt(0) == ' ') {
        c = c.substring(1)
        }
        if (c.indexOf(head) == 0) {
        return c.substring(head.length, c.length)
        }
    }
    return ""
}

function DeleteCookie(name) {
    if( LoadCookie( name ) ) {
        document.cookie = name + "=" +
            ";path=/"+
            ";expires=Thu, 01 Jan 1970 00:00:01 GMT";
    }
}

function PythonClearPrints() {
    Python_textarea.innerHTML = "";
}

function PythonPrint(args) {

    var the_height = parseInt(Python_textarea.style.height, 10);
    var is_at_bottom = (Python_textarea.scrollHeight - Python_textarea.scrollTop < (the_height + 10));

    Python_textarea.innerHTML += args.txt + "&#10;";    

    if (is_at_bottom) {
        Python_textarea.scrollTop = Python_textarea.scrollHeight;
    }

}

var checkbox_DisplayLoopsDuringSearch = document.getElementById('checkbox_DisplayLoopsDuringSearch');
checkbox_DisplayLoopsDuringSearch.addEventListener("change", checkbox_DisplayLoopsDuringSearchHandler, true);

function checkbox_DisplayLoopsDuringSearchHandler(event) {

    var checkbox_DisplayBodiesDuringSearch = document.getElementById('checkbox_DisplayBodiesDuringSearch');

    if (event.currentTarget.checked) {

        checkbox_DisplayBodiesDuringSearch.disabled = "";

    } else {

        checkbox_DisplayBodiesDuringSearch.checked = false;
        checkbox_DisplayBodiesDuringSearch.disabled = "disabled";
        
    }

}

viewport_custom_select = document.getElementById("viewport_custom_select");
viewport_custom_select.addEventListener("input", viewport_custom_select_Handler, true);

function viewport_custom_select_Handler(event) {

    myViewport = document.getElementById("myViewport")

    switch (viewport_custom_select.value) {
        case 'portrait': {
            myViewport.setAttribute('content','width=610')
            break}    
        case 'landscape': {
            myViewport.setAttribute('content','width=1250')
            break}
        case 'disable': {
            myViewport.setAttribute('content','width=device-width')
            break}
    }

}

function ClickStateDiv() {

    if (!SearchIsOnGoing) {
        ChoreoExecuteClick()
    } else {
        ChoreoSearchNextClick()
    }

}

function InitWorkspaceClick() {

    var SetupWorkspaceBtn = document.getElementById("SetupWorkspaceBtn")

    if (!FileSystemAccessSupported) {

        SetupWorkspaceBtn.disabled = "disabled"
        var WorkspaceGalleryContainer = document.getElementById("WorkspaceGalleryContainer")
        WorkspaceGalleryContainer.innerHTML = "The workspace feature is not compatible with your browser."

    }

}

async function ClickSetupWorkspace() {

    try {
        UserWorkspace = await window.showDirectoryPicker({
            id : 'ChoreoUserDir',
            mode: 'readwrite' ,
            startIn: 'documents'
        });

    } catch(e) { // User aborted for instance
        console.log(e);
        return
    }

    WorkspaceIsSetUp = true

    var ReloadWorkspaceBtn = document.getElementById("ReloadWorkspaceBtn")
    ReloadWorkspaceBtn.disabled = ""

    ClickReloadWorkspace()

}

function ClickReloadWorkspace() {

    SaveConfigFile(UserWorkspace)
    
    LoadWorkspaceGallery()

}

async function MakeDirectoryTree_Workspace(cur_directory,cur_treenode,click_callback) {

    var files_here = {}

    for await (const entry of cur_directory.values()) {

        if (entry.kind == "directory") {
            
            var new_node = new TreeNode(entry.name,{expanded:false})
            cur_treenode.addChild(new_node)

            await MakeDirectoryTree_Workspace(await cur_directory.getDirectoryHandle(entry.name),new_node,click_callback)
            
        } else if (entry.kind == "file") {

            for (const ext of [".npy",".json"]) {

                if (entry.name.endsWith(ext)) {
                    
                    basename = entry.name.replace(ext,"")

                    if (files_here[basename] === undefined) {
                        files_here[basename] = {}
                    }

                    files_here[basename][ext] = entry
                    
                } 

            } 

        }

    }

    for (const basename in files_here) {

        if ((!(files_here[basename]['.npy'] === undefined)) && (!(files_here[basename]['.json'] === undefined))) {

            var new_node = new TreeNode(basename,{expanded:false})

            new_node.on("click", (e,node)  => click_callback(basename,files_here[basename]['.npy'],files_here[basename]['.json']));

            cur_treenode.addChild(new_node)

        }
        
    }

}

async function LoadWorkspaceGallery() {

    if (WorkspaceIsSetUp) {
        
        var WorkspaceTree = new TreeNode(UserWorkspace.name,{expanded:true})
        await MakeDirectoryTree_Workspace(UserWorkspace,WorkspaceTree,PlayFileFromDisk)

        var WorkspaceView = new TreeView(WorkspaceTree, "#WorkspaceGalleryContainer",{leaf_icon:" ",parent_icon:" ",show_root:false})

        var WorkspaceTree_Target = new TreeNode("Workspace",{expanded:true})
        await MakeDirectoryTree_Workspace(UserWorkspace,WorkspaceTree_Target,LoadTargetFileFromDisk)

        Target_Tree = new TreeNode("Target_Tree",{expanded:true})
        Target_Tree.addChild(DefaultTree_Target)
        Target_Tree.addChild(WorkspaceTree_Target)

        var Target_TreeView = new TreeView(Target_Tree, "#TreeSlowContainer",{leaf_icon:" ",parent_icon:" ",show_root:false})

    }

}

function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const fr = new FileReader();
        fr.onerror = reject;
        fr.onload = () => {
            resolve(fr.result);
        }
        fr.readAsText(file);
    });
}

function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
        const fr = new FileReader();
        fr.onerror = reject;
        fr.onload = () => {
            resolve(fr.result);
        }
        fr.readAsArrayBuffer(file);
    });
}

async function PlayFileFromDisk(name,npy_file,json_file) {

    var displayCanvas = document.getElementById("displayCanvas")
    var wasrunning = running
    if (running) {
        var event = new Event('StopAnimationFromOutsideCanvas')
        displayCanvas.dispatchEvent(event)
    }

    SolName = name
    
    var RotSlider = $("#RotSlider").data("roundSlider");
    RotSlider.setValue(0);

    const PlotInfoFile = await json_file.getFile()
    PlotInfo = JSON.parse(await readFileAsText(PlotInfoFile))
    UpdateNowPlaying()

    let npyjs_obj = new npyjs()
    const PosFile = await npy_file.getFile()
    Pos = npyjs_obj.parse( await readFileAsArrayBuffer(PosFile))

    Max_PathLength = PlotInfo["Max_PathLength"]
    
    var event = new Event("CompleteSetOrbitFromOutsideCanvas")
    displayCanvas.dispatchEvent(event)
    
    if (wasrunning) {
        var event = new Event('StartAnimationFromOutsideCanvas')
        displayCanvas.dispatchEvent(event)
    }

}

function LoadTargetFileFromDisk(name,npy_file,json_file) {

    UpdateCurrentTarget(name,3)



}

async function PlayFileFromRemote(name,npy_file,json_file) {

    var displayCanvas = document.getElementById("displayCanvas")
    var wasrunning = running
    if (running) {
        var event = new Event('StopAnimationFromOutsideCanvas')
        displayCanvas.dispatchEvent(event)
    }

    npyjs_obj = new npyjs()

    let finished_json = fetch(json_file,Gallery_cache_behavior)
        .then(response => response.text())
        .then(data => {
            PlotInfo = JSON.parse(data)
        })

    let finished_npy = 
        npyjs_obj.load(npy_file)
        .then((res) => {
            Pos = res
        });

    SolName = name

    var RotSlider = $("#RotSlider").data("roundSlider")
    RotSlider.setValue(0)

    await Promise.all([finished_npy ,finished_json ])

    UpdateNowPlaying()
    Max_PathLength = PlotInfo["Max_PathLength"]

    var event = new Event("CompleteSetOrbitFromOutsideCanvas")
    displayCanvas.dispatchEvent(event)

    if (wasrunning) {
        var event = new Event('StartAnimationFromOutsideCanvas')
        displayCanvas.dispatchEvent(event)
    }

}

async function LoadTargetFileFromRemote(name,npy_file,json_file) {

    var The_PlotInfo,The_Pos

    let finished_json = fetch(json_file,Gallery_cache_behavior)
        .then(response => response.text())
        .then(data => {
            The_PlotInfo = JSON.parse(data)
        })

    let finished_npy = 
        npyjs_obj.load(npy_file)
        .then((res) => {
            The_Pos = res
        });

    await finished_json

    UpdateCurrentTarget(name,The_PlotInfo["nloop"])

    if (Target_current_type == "slow") {

        TargetSlow_PlotInfo = The_PlotInfo
        
        await finished_npy
        TargetSlow_Pos = The_Pos

    } else if (Target_current_type == "fast") {

        TargetFast_PlotInfoList[Target_current_id] = The_PlotInfo
        
        await finished_npy
        TargetFast_PosList[Target_current_id] = The_Pos

        TargetFast_LoadedList[Target_current_id] = true
    }

}

function UpdateCurrentTarget(name,nfast) {

    [Target_current_name , Target_current_TreeContainer ] =  ReturnCurrentNameAndTreeContainer()

    Target_current_name.innerHTML = name
    Target_current_TreeContainer.classList.remove('show')

    if (Target_current_type == "slow") {

        var AllTargetFast = document.getElementsByClassName("TargetFast")

        for (var i = 0; i < AllTargetFast.length; i++) {
            AllTargetFast[i].remove()
        }

        var table = document.getElementById('table_targets')

        var i_start_fast = 6
        var i_end = table.rows.length

        for (var i = i_start_fast; i < i_end; i++) {
            table.rows[i_start_fast].remove()
        }

        for (var i = 0; i < nfast; i++) {

            var row = table.insertRow(2*i + i_start_fast)

            var newcell = row.insertCell(0)
            newcell.style.borderStyle = 'none'

            var div = document.createElement('button')
            div.classList.add("TargetFast","TargetToggle","w3-button","w3-light-grey","w3-hover-pale-red","dropdown_btn")
            div.innerHTML = "<b>Choose fast solution "+i.toString()+"</b>"
            div.id = "ChooseFastSolutionBtn"+i.toString()
            div.disabled = ""
            div.onclick = ((j) => ( () => ClickChooseFastSolution(j) ))(i)
            newcell.appendChild(div)
            
            var div = document.createElement('div')
            div.classList.add("TargetFast","dropdown-content")
            div.id = "TreeFastContainer"+i.toString()
            var Target_TreeView = new TreeView(Target_Tree, div,{leaf_icon:" ",parent_icon:" ",show_root:false})

            newcell.appendChild(div)

            var newcell = row.insertCell(1)
            newcell.style.borderStyle = 'none'

            var div = document.createElement('div')
            div.classList.add("TargetFast","TargetToggle")
            div.id = "FastSolution_name"+i.toString()
            newcell.appendChild(div)


            var row = table.insertRow(2*i+1 + i_start_fast)

            var newcell = row.insertCell(0)
            newcell.style.borderStyle = 'none'
            var div = document.createElement('label')
            div.classList.add("TargetFast","TargetToggle")
            div.innerHTML = "Fast period multiplier "+i.toString()
            div.for = "input_nT_fast"+i.toString()
            newcell.appendChild(div)


            var newcell = row.insertCell(1)
            newcell.style.borderStyle = 'none'
            var div = document.createElement('input')
            div.classList.add("TargetFast","TargetToggle")
            div.id = "input_nT_fast"+i.toString()
            div.type = "number"
            div.value = 1
            div.min = 1
            div.style.width="50px"

            newcell.appendChild(div)

        }

        TargetSlow_Loaded = true

        TargetFast_PlotInfoList = new Array(nfast)
        TargetFast_PosList = new Array(nfast)
        TargetFast_LoadedList = []

        for (var i = 0; i < nfast; i++) {

            TargetFast_LoadedList.push(false)

        }

    }

}

function MakeDirectoryTree_DefaultGallery(cur_directory,cur_treenode,click_callback) {

    for (const the_dir of cur_directory.dirs) { 

        var new_node = new TreeNode(the_dir.name,{expanded:false})
        cur_treenode.addChild(new_node)

        MakeDirectoryTree_DefaultGallery(the_dir,new_node,click_callback)

    }

    for (const basename in cur_directory.files) {

        var new_node = new TreeNode(basename,{expanded:false})
        cur_treenode.addChild(new_node)

        new_node.on("click", (e,node)  => click_callback(basename,cur_directory.files[basename]['.npy'],cur_directory.files[basename]['.json']));

    }

}

async function LoadDefaultGallery() {
			
    var gallery_filename = "gallery_descriptor.json"

    await fetch(gallery_filename,Gallery_cache_behavior)
        .then(response => response.text())
        .then(data => {
            DefaultGallery_description = JSON.parse(data);
        })
        
    var DefaultTree = new TreeNode(DefaultGallery_description.name,{expanded:true})
    MakeDirectoryTree_DefaultGallery(DefaultGallery_description,DefaultTree,PlayFileFromRemote)

    var search_leaf = DefaultTree
    while (!search_leaf.isLeaf()) {
        search_leaf.setExpanded(true)
        search_leaf = search_leaf.getChildren()[0]
    }
    search_leaf.setEnabled(true)
    search_leaf.setSelected(true)

    var DefaultTreeView = new TreeView(DefaultTree, "#DefaultGalleryContainer",{leaf_icon:" ",parent_icon:" ",show_root:false})
    
    await search_leaf.getListener("click")()

    DefaultTree_Target = new TreeNode("Gallery",{expanded:true})
    MakeDirectoryTree_DefaultGallery(DefaultGallery_description,DefaultTree_Target,LoadTargetFileFromRemote)

    Target_Tree = DefaultTree_Target

    var Target_TreeView = new TreeView(Target_Tree, "#TreeSlowContainer",{leaf_icon:" ",parent_icon:" ",show_root:false})

}

function FormatMasses(the_mass){ return parseFloat(the_mass.toPrecision(3)).toString() }

function UpdateNowPlaying(SearchOnGoing=false) {

    var NP_name = document.getElementById("NP_name")
    var NP_nbody = document.getElementById("NP_nbody")
    var NP_nloop = document.getElementById("NP_nloop")
    var NP_mass = document.getElementById("NP_mass")

    var NP_Newton_Error = document.getElementById("NP_Newton_Error")
    var NP_n_Fourier = document.getElementById("NP_n_Fourier")
    
    nloop = PlotInfo["nloop"]
    loop_mass = FormatMasses(PlotInfo["mass"][PlotInfo["Targets"][0][0]])
    for (var il = 1 ; il < nloop; il++) {
        loop_mass = loop_mass + ", " + FormatMasses(PlotInfo["mass"][PlotInfo["Targets"][il][0]])
    }

    NP_name.innerHTML = SolName
    NP_nbody.innerHTML = PlotInfo["nbody"].toString()
    NP_nloop.innerHTML = nloop.toString()
    NP_mass.innerHTML = loop_mass

    if (SearchOnGoing) {

        NP_Newton_Error.innerHTML = "Search in progress"
        NP_n_Fourier.innerHTML = "Search in progress"

    } else {

        NP_Newton_Error.innerHTML = PlotInfo["Newton_Error"].toExponential(2)
        NP_n_Fourier.innerHTML = PlotInfo["n_Fourier"].toString()

    }

}

function ClickChooseSlowSolution() {

    Target_current_type = "slow"
    Target_current_id = 0

    ClickChooseTargetSolution()

}

function ClickChooseFastSolution(i) {

    Target_current_type = "fast"
    Target_current_id = i

    ClickChooseTargetSolution()

}

function ReturnCurrentNameAndTreeContainer() {

    var Target_current_name_id
    var Target_current_TreeContainer_id
    if (Target_current_type == "slow") {
        Target_current_name_id = "SlowSolution_name"
        Target_current_TreeContainer_id = "TreeSlowContainer"
    } else if (Target_current_type == "fast") {
        Target_current_name_id = "FastSolution_name"+Target_current_id.toString()
        Target_current_TreeContainer_id = "TreeFastContainer"+Target_current_id.toString()
    }

    var Target_current_name = document.getElementById(Target_current_name_id)
    var Target_current_TreeContainer = document.getElementById(Target_current_TreeContainer_id)

    return [Target_current_name , Target_current_TreeContainer ]

}

function ClickChooseTargetSolution() {

    [Target_current_name , Target_current_TreeContainer ] =  ReturnCurrentNameAndTreeContainer()

    OpenMe = !Target_current_TreeContainer.classList.contains('show')

    var AllDropdowns = document.getElementsByClassName("dropdown-content");

    for (var i = 0; i < AllDropdowns.length; i++) {
        if (AllDropdowns[i].classList.contains('show')) {
            AllDropdowns[i].classList.remove('show')
        }
    }

    if (OpenMe) {
        Target_current_TreeContainer.classList.add('show')
    }

}

function checkbox_EnableTargets_Handler(event) {

    var AllDropdowns = document.getElementsByClassName("dropdown-content");

    for (var i = 0; i < AllDropdowns.length; i++) {
        if (AllDropdowns[i].classList.contains('show')) {
            AllDropdowns[i].classList.remove('show')
        }
    }

    var AllTargetToggle = document.getElementsByClassName("TargetToggle");

    for (var i = 0; i < AllTargetToggle.length; i++) {
        AllTargetToggle[i].disabled = ! AllTargetToggle[i].disabled // Toggling
    }

}