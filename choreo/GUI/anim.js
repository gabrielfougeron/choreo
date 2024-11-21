var Gallery_cache_behavior = {}

var DefaultGallery_description

var UserWorkspace
var WorkspaceIsSetUp = false

var SolName 
var Pos 
var PlotInfo

var running = false

var xMin=0., xMax=1., yMin=0., yMax=1.
var Max_PathLength = 1.

var xPixRate, yPixRate
var center_x,center_y
var CurrentMax_PartRelSize = 1.

var displayWidth, displayHeight
var trajectoriesOn

var FPS_estimation = 60
var Do_Limit_FPS = false
var FPS_limit = 120
var Elapsed_Time_During_Animation = 0
var n_valid_dt_animation = 0

var real_period_estimation = 1.

var Last_UpdateFPSDisplay = 0
var UpdateFPSDisplay_freq = 5

var n_color = 0
var colorLookup_init = [
	"#50ce4d", // Moderate Lime Green
	"#ff7006", // Vivid Orange
	"#a253c4", // Moderate Violet
	"#ef1010", // Vivid Red
	"#25b5bc", // Strong Cyan
	"#E86A96", // Soft Pink
	"#edc832", // Bright Yellow
	"#ad6530", // Dark Orange [Brown tone]
	"#00773f", // Dark cyan - lime green 
	"#d6d6d6", // Light gray
]

var defaultParticleColor = "#50ce4d"
var defaultTrailColor = "#3c9a39"
var FallbackTrailColor = "#d5d5d5"

// Particle radius
var min_base_particle_size = 1.
var max_base_particle_size = 10.
var base_particle_size = 6.

// with of particle trail
var min_base_trailWidth = 0.2
var max_base_trailWidth = 5.
var base_trailWidth = 2

// Vanish speed
var min_base_trail_vanish_length = 0.1
var max_base_trail_vanish_length = 10.
var base_trail_vanish_length = 1.
var trail_vanish_length_mul = 0.002

var DoScaleSizeWithMass = true
var DoTrailVanish = true
var LastFadeTime = 0

var SearchIsOnGoing = false

var FadeInvFrequency

var	colorLookup = colorLookup_init

var AllPosFilenames = []
var AllPlotInfoFilenames = []
var AllGalleryNames = []

var DefaultTree
var WorkspaceTree

var DefaultTree_Target
var Target_Tree

var Target_current_type
var Target_current_id

var TargetSlow_PlotInfo
var TargetSlow_Pos
var TargetSlow_Loaded = false

var TargetFast_PlotInfoList
var TargetFast_PosList
var	TargetFast_LoadedList

var trailColorLookup

var mainCanvas
var mainContext

var trailLayerCanvas
var trailLayerContext

var particleLayerCanvas
var particleLayerContext

var FileSystemAccessSupported = ('showOpenFilePicker' in window)

var BodyGraph
var PreviousInputValueNbody = 3
var MassArray = [1.]
var ChargeArray = [1.]
var LoopTargets = []

const { createFFmpeg , fetchFile  } = FFmpeg
const ffmpeg = createFFmpeg()

function AjaxGet(foldername){ return $.ajax({ url: foldername})}

function GetPlotInfoChoreoVersion() {

	if ("choreo_version" in PlotInfo) {
		return PlotInfo["choreo_version"]
	} else {
		return "legacy"
	}

}

function GetFileBaseExt(filename) {

	var dotidx = filename.lastIndexOf('.')+1;

	var base = filename.substring(0,dotidx-1);
	var ext = filename.substring(dotidx-1, filename.length);

	return [base,ext]

}

async function ListFilesInFolder(folder){

	var FilesArray = [];

	$.ajax({
		url: folder,
		success: function(data){
			$(data).find("li > a").each(function(){
				FilesArray.push(folder+this.innerHTML);
			})
		}           
	});

	return FilesArray;

}

function windowLoadHandler() {
	canvasApp();
}

function setPlotWindow(windowObject) {

	var extend = 0.02 + (CurrentMax_PartRelSize-1.)*0.0155 ; // Magic numbers

	var xinf = windowObject.xMin - extend*(windowObject.xMax-windowObject.xMin);
	var xsup = windowObject.xMax + extend*(windowObject.xMax-windowObject.xMin);
	
	var yinf = windowObject.yMin - extend*(windowObject.yMax-windowObject.yMin);
	var ysup = windowObject.yMax + extend*(windowObject.yMax-windowObject.yMin);
	
	var hside = Math.max(xsup-xinf,ysup-yinf)/2

	center_x = (xinf+xsup)/2
	center_y = (yinf+ysup)/2

	xMin = center_x - hside
	xMax = center_x + hside

	yMin = center_y - hside
	yMax = center_y + hside

	xPixRate = displayWidth/(xMax - xMin);
	yPixRate = displayHeight/(yMin - yMax);

}

function setColorLookupList() {
		
	trailColorLookup = [];
	
	//darkening colors for trails
	var i,r,g,b,colorString,c, newColor,newColorString;
	var len = colorLookup.length;
	var darkenFactor = 0.75;
	for (i = 0; i < len; i++) {
		colorString = colorLookup[i];
		colorString = colorString.replace("#", "");
		c = parseInt(colorString,16);
		r = (c & (255 << 16)) >> 16;
		g = (c & (255 << 8)) >> 8;
		b = (c & 255); 
		
		r = Math.floor(r*darkenFactor);
		g = Math.floor(g*darkenFactor);
		b = Math.floor(b*darkenFactor);
					
		newColor = (r << 16) | (g << 8) | b;
		
		newColorString = newColor.toString(16);
		while (newColorString.length < 6) {
			newColorString = "0" + newColorString;
		}
		newColorString = "#" + newColorString;
		trailColorLookup.push(newColorString);
	}

}

function canvasApp() {
		
	var particles;

	var tInc;

	var time = 0;
	var Lasttime = time;

	var bgColor = "#F1F1F1";
	var request;

	var fadeScreenColor = "rgba(241,241,241,0.01)";
	
	var staticOrbitDrawPointsX;
	var staticOrbitDrawPointsY;
	
	var endPixX;
	var endPixY;

	var GlobalRot_angle = 0.;
	var GlobalRot = [[1.,0.],[0.,1.]];

	mainCanvas = document.getElementById("mainCanvas");
	mainContext = mainCanvas.getContext("2d");

	trailLayerCanvas = document.getElementById("trailLayerCanvas");
	trailLayerContext = trailLayerCanvas.getContext("2d");
	trailLayerContext.lineCap = "round";
	trailLayerCanvas.addEventListener("FinalizeSetOrbitFromOutsideCanvas"	, FinalizeSetOrbitFromOutsideCanvasHandler	, true)
	trailLayerCanvas.addEventListener("FinalizeAndPlayFromOutsideCanvas"	, FinalizeAndPlayFromOutsideCanvasHandler	, true)
	trailLayerCanvas.addEventListener("DisableAnimationFromOutsideCanvas"	, DisableAnimationFromOutsideCanvasHandler	, true)
	trailLayerCanvas.addEventListener("EnableAnimationFromOutsideCanvas"	, EnableAnimationFromOutsideCanvasHandler	, true)
	trailLayerCanvas.addEventListener("StartAnimationFromOutsideCanvas"		, StartAnimationFromOutsideCanvasHandler	, true)
	trailLayerCanvas.addEventListener("RemakeParticlesFromOutsideCanvas"	, RemakeParticlesFromOutsideCanvasHandler	, true)
	trailLayerCanvas.addEventListener("ChangeColorsFromOutsideCanvas"		, ChangeColorsFromOutsideCanvasHandler		, true)
	trailLayerCanvas.addEventListener("DrawAllPathsFromOutsideCanvas"		, DrawAllPathsFromOutsideCanvasHandler		, true)
	trailLayerCanvas.addEventListener("CompleteSetOrbitFromOutsideCanvas"	, CompleteSetOrbitFromOutsideCanvasHandler	, true)
	trailLayerCanvas.addEventListener("ErrorFromOutsideCanvas"				, ErrorFromOutsideCanvasHandler				, true)
	
	particleLayerCanvas = document.getElementById("particleLayerCanvas");
	particleLayerContext = particleLayerCanvas.getContext("2d");
	particleLayerCanvas.addEventListener("click", startStopButtonHandler, true);
	
	displayWidth = trailLayerCanvas.width;
	displayHeight = trailLayerCanvas.height;	

	var Min_PartRelSize = 0.5;
	var Max_PartRelSize = 70.;
	
	var Last_Time_since_origin;
	var dt_outlier_ms = 1200;
	var speed_slider_value_init = .5;
	var Time_One_Period_init = 5;
	
	var startStopButton = document.getElementById("startStopButton");
	startStopButton.addEventListener("click", startStopButtonHandler, true);
	
	var trajectoryButton = document.getElementById("trajectoryButton");
	trajectoryButton.addEventListener("click", trajectoryButtonHandler, true);

	var drawTrajButton = document.getElementById("ClearButton");
	drawTrajButton.addEventListener("click", clearScreen, true);

	var speedPlusBtn = document.getElementById("speedPlusBtn");
	speedPlusBtn.addEventListener("click", SpeedPlusClick, true);
	
	var speedMinusBtn = document.getElementById("speedMinusBtn");
	speedMinusBtn.addEventListener("click", SpeedMinusClick, true);

	var speedTxt = document.getElementById("speedTxt");

	// http://paulirish.com/2011/requestanimationframe-for-smart-animating/
	// http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating
	
	// requestAnimationFrame polyfill by Erik Möller
	// fixes from Paul Irish and Tino Zijdel
	
	(function() {
		var the_lastTime = 0;
		var vendors = ['ms', 'moz', 'webkit', 'o'];
		for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
			window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
			window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame']
									|| window[vendors[x]+'CancelRequestAnimationFrame'];
		}
	
		if (!window.requestAnimationFrame)
			window.requestAnimationFrame = function(callback, element) {
				var currTime = new Date().getTime();
				var timeToCall = Math.max(0, 16 - (currTime - the_lastTime));
				var id = window.setTimeout(function() { callback(currTime + timeToCall); },
				timeToCall);
				the_lastTime = currTime + timeToCall;
				return id;
			};
	
		if (!window.cancelAnimationFrame)
			window.cancelAnimationFrame = function(id) {
				clearTimeout(id);
			};
	}());
	
	init();
	
	async function init() {
		
		setColorLookupList()

		trajectoriesOn = true;
		
		//jquery ui elements
		//speed slider
		$("#speedSlider").slider({
		  value: speed_slider_value_init,
		  orientation: "horizontal",
		  range: "min",
		  max: 1,
		  min: 0.2,
		  step: 0.005,
		  input:speedSliderHandler,
		  change:speedSliderHandler,
		  animate: false,
		});		

		//Rotation slider
		$("#RotSlider").roundSlider({
				width: 15,
				radius: 50,
				value: 0,
				keyboardAction: false,
				lineCap: "square",
				startAngle: 90,
				max: 360,
				mouseScrollAction: true,
				step: 5,
				// editableTooltip: false,
				// change:onRotationChange,
				valueChange:onRotationValueChange,
				disabled:false,
			});
		
		// Load the static gallery
		await LoadDefaultGallery()

		startAnimation()

	}

	function Estimate_FPS(Time_since_origin){

		if (Last_Time_since_origin && Time_since_origin)
		{
			var dt_ms = Time_since_origin - Last_Time_since_origin;

			if ( dt_ms < dt_outlier_ms)
			{
				Elapsed_Time_During_Animation += dt_ms/1000;
				n_valid_dt_animation +=1;

				FPS_estimation = n_valid_dt_animation/Elapsed_Time_During_Animation;

			}
		
		}
	
		Last_Time_since_origin = Time_since_origin;	

		setPeriodTime()

	}

	function anim_particles() {
		
		UpdateFPSDisplay();
		onTimer();
	}

	function anim_particles_loop(Time_since_origin){

		if (running) {
			request = requestAnimationFrame(anim_particles_loop)
		}

		if (Do_Limit_FPS) {

			if (anim_schedule_time < Time_since_origin) {
				anim_schedule_time += 1000/FPS_limit 
				Estimate_FPS(Time_since_origin)
				anim_particles()
			}

		} else {
			Estimate_FPS(Time_since_origin)
			anim_particles()
		}

	}
	
	function startAnimation() {
		if (!running) {
			running = true
			startStopButton.textContent = "Stop"
			input_Limit_FPS_Handler()
			anim_particles_loop()
		}
	}

	function stopAnimation() {
		if (running) {
			running = false
			startStopButton.textContent = "Start"
		}
	}
	
	function startStopButtonHandler(e) {

		if (!document.getElementById("startStopButton").disabled) {
			if (running) {
				stopAnimation()
			}
			else {
				startAnimation()
			}
		}
	}

	function trajectoryButtonHandler(e) {
		if (trajectoriesOn) {
			trajectoriesOn = false
			trajectoryButton.textContent ="Draw trails"
			clearScreen()
		}
		else {
			setStartPositions()
			trajectoriesOn = true
			trajectoryButton.textContent = "Hide trails"
		}
	}

	function StopAnimationFromOutsideCanvasHandler(e) {
		stopAnimation()
	}

	function DisableAnimationFromOutsideCanvasHandler(e) {
		clearParticleLayer()
		clearScreen()		
		DisableAnimation()
	}

	function DisableAnimation() {
		document.getElementById("startStopButton").disabled = "disabled";
	}

	function EnableAnimationFromOutsideCanvasHandler(e) {
		EnableAnimation()
	}

	function EnableAnimation() {
		document.getElementById("startStopButton").disabled = ""
	}

	function StartAnimationFromOutsideCanvasHandler(e) {
		startAnimation()
	}

	function FinalizeSetOrbitFromOutsideCanvasHandler(e) {

		clearScreen()
		clearParticleLayer()
		FinalizeSetOrbit(DoDrawParticles = false, DoXMinMax = true, setTinc = false)

		if (document.getElementById('checkbox_DisplayBodiesDuringSearch').checked) {
			startAnimation()
		}

	}

	function CompleteSetOrbitFromOutsideCanvasHandler(e) {
		CompleteSetOrbit()
	}

	function FinalizeAndPlayFromOutsideCanvasHandler(e) {

		if (e.DoClearScreen) {
			clearScreen()

			if (e.ResetRot) {
				var RotSlider = $("#RotSlider").data("roundSlider")
				RotSlider.setValue(0)
			}
			
			if (trajectoriesOn && document.getElementById('checkbox_DisplayLoopOnGalleryLoad').checked){

				if (DoTrailVanish) {
					request = requestAnimationFrame(anim_path_grey)
				} else {
					request = requestAnimationFrame(anim_path)
				}
				
			}

		}
		
		FinalizeSetOrbit(DoDrawParticles = true, DoXMinMax = e.DoXMinMax, setTinc = e.setTinc)
		startAnimation()
		
	}

	function ErrorFromOutsideCanvasHandler(e) {

		stopAnimation()
		clearParticleLayer()

		trailLayerContext.fillStyle = "#FF0000"
		trailLayerContext.fillRect(0,0,displayWidth,displayHeight)
		
	}

	function RemakeParticlesFromOutsideCanvasHandler(e) {
		makeParticles()
		clearScreen()
		clearParticleLayer()
		drawParticles()
	}

	function ChangeColorsFromOutsideCanvasHandler(e) {
		
		setColorLookupList()
		makeParticles()
		clearScreen()
		clearParticleLayer()
		drawParticles()
	}

	function onRotationValueChange(e){

		if ((e.action == 'code') || (e.action == 'change') || (e.action == 'drag')) {

			GlobalRot_angle = e.value * 2* Math.PI / 360.;
			GlobalRot = [
				[ Math.cos(GlobalRot_angle), Math.sin(GlobalRot_angle)],
				[-Math.sin(GlobalRot_angle), Math.cos(GlobalRot_angle)]
			]

			var delta_angle = (e.value - e.preValue)* 2* Math.PI / 360.;
			
			RotateCanvas(trailLayerCanvas,trailLayerContext,delta_angle);
			
			setParticlePositions(time);
			setParticlePositions(time); // dirty hack
			clearParticleLayer();
			drawParticles();

		}
	}

	function RotateCanvas(canvas,ctx,angle){

		var tempCanvas = document.createElement("canvas");
		var tempCtx = tempCanvas.getContext("2d");

		tempCanvas.width = canvas.width;
		tempCanvas.height = canvas.height;
		tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height);

		ctx.save(); //saves the state of canvas
		ctx.fillStyle = bgColor;
		ctx.fillRect(0,0,canvas.width,canvas.height);

		ctx.translate(
			canvas.width * 0.5,
			canvas.height * 0.5
		);
		ctx.rotate(angle);
		ctx.translate(
			-canvas.width * 0.5,
			-canvas.height * 0.5
		);
		ctx.drawImage(tempCanvas,  0, 0, canvas.width, canvas.height);
		ctx.restore();

	};

	function onTimer() {

		if (trajectoriesOn) {
			
			//fade
			trailLayerContext.fillStyle = fadeScreenColor

			
			var nfade = Math.floor(LastFadeTime/FadeInvFrequency)
			if (!DoTrailVanish) {nfade = 0}

			for (var ifade=0; ifade<nfade; ifade++){
				trailLayerContext.fillRect(0,0,displayWidth,displayHeight)
			}

			LastFadeTime = LastFadeTime + tInc - nfade*FadeInvFrequency

		}
		
		//clear particle layer
		clearParticleLayer()
		
		Lasttime = time
		time = (time + tInc) % 1
		
		//update particles
		setParticlePositions(time)
		
		//draw particles
		drawParticles()

		if (trajectoriesOn) {
			if (Pos !== undefined) {
				drawPathPortion(Lasttime,tInc) 
			}
		}

		mainContext.drawImage(trailLayerCanvas, 0, 0);
		mainContext.drawImage(particleLayerCanvas, 0, 0);

	}
	
	function clearScreen() {
		trailLayerContext.fillStyle = bgColor
		trailLayerContext.fillRect(0,0,displayWidth,displayHeight)
	}
	
	function clearParticleLayer() {
		particleLayerContext.clearRect(0,0,displayWidth+1,displayHeight+1)
	}	

	function makeParticles() {

		var ib
		var color
		var trailColor
		var color_id
		var PartRelSize

		particles = new Array(PlotInfo['nbody'])

		CurrentMax_PartRelSize = 0
		
		var color_method_input = document.getElementById("color_method_input")

		var IsLegacy = (GetPlotInfoChoreoVersion() == "legacy")

        for ( var il = 0 ; il < PlotInfo['nloop'] ; il++){

			var nlb =  PlotInfo['loopnb'][il]
			
			for (var ilb = 0 ; ilb < nlb ; ilb++){

				ib = PlotInfo['Targets'][il][ilb]


				if (color_method_input.value == "body") {
					color_id = ib
				} else if (color_method_input.value == "loop") {
					color_id = il
				} else if (color_method_input.value == "loop_id") {
					color_id = ilb
				} else {
					color_id = 0
				}
				
				color = colorLookup[color_id % colorLookup.length];
				trailColor = trailColorLookup[color_id % colorLookup.length];
				
				if (DoScaleSizeWithMass) {
					if (IsLegacy) { // Really useless distinction but hey ... legacy is not forever, right ?
						PartRelSize = Math.sqrt(PlotInfo["mass"][ib])
					} else {
						PartRelSize = Math.sqrt(PlotInfo["bodymass"][ib])
					}
				} else {
					PartRelSize = 1.
				}

				PartRelSize = Math.min(PartRelSize, Max_PartRelSize)
				PartRelSize = Math.max(PartRelSize, Min_PartRelSize)
				// Min/max ?
				
				CurrentMax_PartRelSize = Math.max(CurrentMax_PartRelSize,PartRelSize)
						
				particles[ib] = {
						x: 0.					,
						y: 0.					,
						lastX: 0.				,
						lastY: 0.				,
						color: color			,
						trailColor: trailColor	,
						PartRelSize: PartRelSize,
				}

			}
		}

		setParticlePositions(time);
		resetLastPositions();
	}
	
	function resetLastPositions() {
		//set initial last positions
		for (i = 0; i<PlotInfo["nbody"]; i++) {
			particles[i].lastX = particles[i].x;
			particles[i].lastY = particles[i].y;
		}
	}
	
	function setStartPositions() {
		var pixX;
		var pixY;
		var j;
		endPixX = [];
		endPixY = [];
		staticOrbitDrawPointsX = [];
		staticOrbitDrawPointsY = [];
		for (i = 0; i<PlotInfo["nbody"]; i++) {
			j = (i + 1) % PlotInfo["nbody"];
			pixX = xPixRate*(particles[j].x - xMin);
			pixY = yPixRate*(particles[j].y - yMax);
			endPixX.push(pixX);
			endPixY.push(pixY);
			staticOrbitDrawPointsX.push(xPixRate*(particles[i].x - xMin));
			staticOrbitDrawPointsY.push(yPixRate*(particles[i].y - yMax));
		}
	}

	function anim_path(Time_since_origin){

		clearScreen()
		DrawAllPaths("particle")

	}

	function anim_path_grey(Time_since_origin){

		clearScreen()
		DrawAllPaths(FallbackTrailColor)

	}

	function DrawAllPathsFromOutsideCanvasHandler() {

		request = requestAnimationFrame(anim_path)

	}

	function DrawAllPaths(color) {
		if (GetPlotInfoChoreoVersion() == "legacy") {
			DrawAllPaths_legacy(color)
		} else {
			DrawAllPaths_new(color)
		}
	}

	function drawPathPortion(lasttime, tInc) {

		if (GetPlotInfoChoreoVersion() == "legacy") {
			drawPathPortion_legacy(lasttime, tInc)
		} else {
			drawPathPortion_new(lasttime, tInc)
		}

	}

	function setParticlePositions(t) {
		if (GetPlotInfoChoreoVersion() == "legacy") {
			setParticlePositions_legacy(t)
		} else {
			setParticlePositions_new(t)
		}
	}
	
	function anim_path_grey(Time_since_origin){

		clearScreen()
		DrawAllPaths(FallbackTrailColor)

	}
	
	function anim_path_grey(Time_since_origin){

		clearScreen()
		DrawAllPaths(FallbackTrailColor)

	}

	function DrawAllPaths_legacy(color="particle") {

		var n_pos = Pos.shape[2];

		var xl,yl;
		var x,y;
		var Pixx,Pixy;

		var il,ib,ilb,nlb;
		var p;

        for ( il = 0 ; il < PlotInfo['nloop'] ; il++){

			nlb =  PlotInfo['loopnb'][il];
			
			for (ilb = 0 ; ilb < nlb ; ilb++){

				if (PlotInfo['RequiresLoopDispUn'][il][ilb]) {

					ib = PlotInfo['Targets'][il][ilb];
					p = particles[ib];
					trailLayerContext.lineWidth = p.PartRelSize * base_trailWidth ;

					if (color == "particle") {
						trailLayerContext.strokeStyle = p.trailColor;
					} else {
						trailLayerContext.strokeStyle = color;
					}

					// Super ugly
					xl = Pos.data[  2*il    * n_pos] ;
					yl = Pos.data[ (2*il+1) * n_pos] ;

					x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
					y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;

					Pixx = xPixRate*(x - xMin) ;
					Pixy = yPixRate*(y - yMax) ;

					trailLayerContext.beginPath();
					trailLayerContext.moveTo(Pixx, Pixy);

					for (i_pos = 1 ; i_pos < n_pos ; i_pos++){

						// Super ugly
						xl = Pos.data[ i_pos +  2*il    * n_pos] ;
						yl = Pos.data[ i_pos + (2*il+1) * n_pos] ;

						x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
						y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;
		
						Pixx = xPixRate*(x - xMin) ;
						Pixy = yPixRate*(y - yMax) ;

						trailLayerContext.lineTo(Pixx, Pixy);

					}

					// Super ugly
					xl = Pos.data[  2*il    * n_pos] ;
					yl = Pos.data[ (2*il+1) * n_pos] ;

					x = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yl ;
					y = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xl + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yl ;

					Pixx = xPixRate*(x - xMin) ;
					Pixy = yPixRate*(y - yMax) ;

					trailLayerContext.lineTo(Pixx, Pixy);
					trailLayerContext.stroke();

				}

			}

		}

	}

	function DrawAllPaths_new(color="particle") {

		var xl,yl
		var x,y
		var Pixx,Pixy

		var p
		var io, ib, iint, isegm

		var RotMat

		var segm_store = Pos.shape[1]
		var segm_size = PlotInfo["segm_size"]
		var nint_min = PlotInfo['nint_min']

		for (ib=0; ib<PlotInfo['nbody']; ib++) {

			for (iint=0; iint<nint_min; iint++) {

				if (PlotInfo['SegmRequiresDisp'][ib][iint]) {
					
					isegm = PlotInfo['bodysegm'][ib][iint]
					i = segm_store * isegm * 2

					p = particles[ib];
					trailLayerContext.lineWidth = p.PartRelSize * base_trailWidth 

					if (color == "particle") {
						trailLayerContext.strokeStyle = p.trailColor
					} else {
						trailLayerContext.strokeStyle = color
					}

					RotMat = PlotInfo['InterSegmSpaceRot'][ib][iint]

					// Super ugly
					xl = Pos.data[i]
					i += 1
					yl = Pos.data[i] 
					i += 1

					x = RotMat[0][0] * xl + RotMat[0][1] * yl 
					y = RotMat[1][0] * xl + RotMat[1][1] * yl 

					Pixx = xPixRate*(x - xMin) 
					Pixy = yPixRate*(y - yMax) 

					trailLayerContext.beginPath()
					trailLayerContext.moveTo(Pixx, Pixy)

					for (i_pos = 1 ; i_pos < segm_store ; i_pos++){

						// Super ugly
						xl = Pos.data[i] 
						i += 1
						yl = Pos.data[i] 
						i += 1

						x = RotMat[0][0] * xl + RotMat[0][1] * yl 
						y = RotMat[1][0] * xl + RotMat[1][1] * yl 
		
						Pixx = xPixRate*(x - xMin) 
						Pixy = yPixRate*(y - yMax) 

						trailLayerContext.lineTo(Pixx, Pixy)

					}

					if (segm_size == segm_store){
						jint = (iint + 1) % nint_min

						RotMat = PlotInfo['InterSegmSpaceRot'][ib][jint]
						isegm = PlotInfo['bodysegm'][ib][jint]
						i = segm_store * isegm * 2

						// Super ugly
						xl = Pos.data[i]
						i += 1
						yl = Pos.data[i] 
						i += 1

						x = RotMat[0][0] * xl + RotMat[0][1] * yl 
						y = RotMat[1][0] * xl + RotMat[1][1] * yl 

						Pixx = xPixRate*(x - xMin) 
						Pixy = yPixRate*(y - yMax) 

						trailLayerContext.lineTo(Pixx, Pixy)

					}
					
					trailLayerContext.stroke()

				}
			}

		}

	}

	function drawParticles() {
		var i;
		var len;
		var pixX;
		var pixY;
		var p;
		
		len = particles.length;
		for (i = 0; i < len; i++) {
			p = particles[i];
			pixX = xPixRate*(p.x - xMin);
			pixY = yPixRate*(p.y - yMax);
			lastPixX = xPixRate*(p.lastX - xMin);
			lastPixY = yPixRate*(p.lastY - yMax);
						
			var size = p.PartRelSize * base_particle_size

			//particle
			particleLayerContext.strokeStyle = "rgba(0,0,0,0.5)"
			particleLayerContext.lineWidth = 2;
			particleLayerContext.fillStyle = p.color;
			particleLayerContext.beginPath();
			particleLayerContext.arc(pixX,pixY,size,0,Math.PI*2,false);
			particleLayerContext.closePath();
			particleLayerContext.fill();
			particleLayerContext.stroke();

		}

	}

	function drawPathPortion_legacy(lasttime, tInc) {

		var n_pos = Pos.shape[2]

		var xi=0,yi=0
		var xrot,yrot
		var xrot_glob,yrot_glob
		var p

		var il,ib,ilb,nlb
		var tb_beg, tb_end

		var i,di
		var ip,ip_beg,ip_end

        for ( il = 0 ; il < PlotInfo['nloop'] ; il++){

			nlb =  PlotInfo['loopnb'][il]
			
			for (ilb = 0 ; ilb < nlb ; ilb++){

				ib = PlotInfo['Targets'][il][ilb]
				p = particles[ib]
				trailLayerContext.strokeStyle = p.trailColor
				trailLayerContext.lineWidth = p.PartRelSize * base_trailWidth 

				PixX = xPixRate*(p.lastX - xMin)
				PixY = yPixRate*(p.lastY - yMax)
				trailLayerContext.beginPath()
				trailLayerContext.moveTo(PixX,PixY)

				tb_beg = ( PlotInfo['TimeRevsUn'][il][ilb] * (lasttime - PlotInfo['TimeShiftNumUn'][il][ilb] / PlotInfo['TimeShiftDenUn'][il][ilb]) +1)
				ip_beg = (Math.floor(tb_beg*n_pos)+1)

				tb_end = ( PlotInfo['TimeRevsUn'][il][ilb] * ((lasttime+tInc) - PlotInfo['TimeShiftNumUn'][il][ilb] / PlotInfo['TimeShiftDenUn'][il][ilb]) +1)
				ip_end = (Math.floor(tb_end*n_pos)+1)

				di = PlotInfo['TimeRevsUn'][il][ilb]

				for (i = ip_beg ; i != ip_end ; i += di){ 

					ip = (((i%n_pos) + n_pos) % n_pos)

					// Super ugly
					xi = Pos.data[ ip +  2*il    * n_pos] 
					yi = Pos.data[ ip + (2*il+1) * n_pos] 

					xrot = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xi + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * yi - center_x
					yrot = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xi + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * yi - center_y

					xrot_glob = center_x + GlobalRot[0][0] * xrot + GlobalRot[0][1] * yrot
					yrot_glob = center_y + GlobalRot[1][0] * xrot + GlobalRot[1][1] * yrot

					PixX = xPixRate*(xrot_glob - xMin)
					PixY = yPixRate*(yrot_glob - yMax)

					trailLayerContext.lineTo(PixX, PixY)

				}

				PixX = xPixRate*(p.x - xMin);
				PixY = yPixRate*(p.y - yMax);

				trailLayerContext.lineTo(PixX, PixY)
				trailLayerContext.stroke()

			}
		
		}

	}

	function drawPathPortion_new(tbeg, tInc) {

		var ib
		var PixX, PixY

		var segm_store = Pos.shape[1]
		if (PlotInfo["segm_size"] == PlotInfo["segm_store"]){
			var segm_size = segm_store
		} else {
			var segm_size = segm_store - 1
		}
		
		var tmul_beg = tbeg*PlotInfo['nint_min']
		var iint_min_beg = Math.floor(tmul_beg)
		var trem_beg = tmul_beg - iint_min_beg
		var irem_beg = Math.floor(trem_beg*segm_size)
		
		var tend = tbeg + tInc
		var tmul_end = tend*PlotInfo['nint_min']
		var iint_min_end = Math.floor(tmul_end)
		var trem_end = tmul_end - iint_min_end
		var irem_end = Math.floor(trem_end*segm_size)

		var iint_min_loop, iint_min

		var ip, ip_beg, ip_end
		var io, iip
		var xl, yl
		var TimeRev
		var RotMat

		for (ib=0; ib<PlotInfo['nbody']; ib++) {

			p = particles[ib]
			trailLayerContext.strokeStyle = p.trailColor
			trailLayerContext.lineWidth = p.PartRelSize * base_trailWidth 
			PixX = xPixRate*(p.lastX - xMin)
			PixY = yPixRate*(p.lastY - yMax)
			trailLayerContext.beginPath()
			trailLayerContext.moveTo(PixX, PixY)

			for (iint_min_loop = iint_min_beg; iint_min_loop<=iint_min_end; iint_min_loop++) {

				iint_min = iint_min_loop % PlotInfo['nint_min']

				RotMat = PlotInfo['InterSegmSpaceRot'][ib][iint_min]
				isegm = PlotInfo['bodysegm'][ib][iint_min]
				io = segm_store * isegm * 2

				ip_beg = 0
				ip_end = segm_size

				if (iint_min_loop == iint_min_beg) {
					ip_beg = irem_beg
				}
				if (iint_min_loop == iint_min_end) {
					ip_end = irem_end
				}
				
				TimeRev = (PlotInfo["InterSegmTimeRev"][ib][iint_min] > 0)

				for (ip = ip_beg; ip < ip_end; ip++){

					if (TimeRev) {
						iip = io + 2*ip
					} else {
						iip = io + 2*(segm_size-ip)
					}

					xl = RotMat[0][0] * Pos.data[ iip ] + RotMat[0][1] * Pos.data[ iip + 1 ] - center_x
					yl = RotMat[1][0] * Pos.data[ iip ] + RotMat[1][1] * Pos.data[ iip + 1 ] - center_y
		
					xrot_glob = center_x + GlobalRot[0][0] * xl + GlobalRot[0][1] * yl
					yrot_glob = center_y + GlobalRot[1][0] * xl + GlobalRot[1][1] * yl

					PixX = xPixRate*(xrot_glob - xMin)
					PixY = yPixRate*(yrot_glob - yMax)

					trailLayerContext.lineTo(PixX, PixY)

				}

			}

			PixX = xPixRate*(p.x - xMin);
			PixY = yPixRate*(p.y - yMax);

			trailLayerContext.lineTo(PixX, PixY)
			trailLayerContext.stroke()

		}

	}
	
	function setParticlePositions_legacy(t) {

		var n_pos = Pos.shape[2];

		var xlm=0,xlp=0,ylm=0,ylp=0;
		var xmid,ymid;
		var xrot,yrot;
		var xrot_glob,yrot_glob;

		var il,ib,ilb,nlb;
		var tb;

		var im,ip,tbn,trem;

        for ( il = 0 ; il < PlotInfo['nloop'] ; il++){

			nlb =  PlotInfo['loopnb'][il];
			
			for (ilb = 0 ; ilb < nlb ; ilb++){

				ib = PlotInfo['Targets'][il][ilb];

				tb = ( PlotInfo['TimeRevsUn'][il][ilb] * (t - PlotInfo['TimeShiftNumUn'][il][ilb] / PlotInfo['TimeShiftDenUn'][il][ilb]) +1) % 1;

				tbn = tb*n_pos;
				
				im = Math.floor(tbn);
				ip = (im+1) % n_pos;
				
				trem = tbn - im;

				// Super ugly
				xlm = Pos.data[ im +  2*il    * n_pos] ;
				ylm = Pos.data[ im + (2*il+1) * n_pos] ;
				xlp = Pos.data[ ip +  2*il    * n_pos] ;
				ylp = Pos.data[ ip + (2*il+1) * n_pos] ;
			
				xmid = (1-trem) * xlm + trem * xlp;
				ymid = (1-trem) * ylm + trem * ylp;

				xrot = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xmid + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * ymid ;
				yrot = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xmid + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * ymid ;

				xrot = xrot - center_x;
				yrot = yrot - center_y;

				xrot_glob = center_x + GlobalRot[0][0] * xrot + GlobalRot[0][1] * yrot
				yrot_glob = center_y + GlobalRot[1][0] * xrot + GlobalRot[1][1] * yrot

				particles[ib].lastX = particles[ib].x;
				particles[ib].lastY = particles[ib].y;
				particles[ib].x = xrot_glob;
				particles[ib].y = yrot_glob;
				
			}
		
		}

	}

	function setParticlePositions_new(t) {

		var segm_store = Pos.shape[1]
		if (PlotInfo["segm_size"] == PlotInfo["segm_store"]){
			var segm_size = segm_store
		} else {
			var segm_size = segm_store - 1
		}

		var p
		var iom, iop, ib, im, ip, isegmm, isegmp
		var iim, iip
		var iint_minm, iint_minp
		var tmul, trem

		tmul = t*PlotInfo['nint_min']
		iint_minm = Math.floor(tmul)
		trem = tmul - iint_minm

		tmul = trem*segm_size
		im = Math.floor(tmul)
		trem = tmul - im

		ip = im+1
		if (ip >= segm_store) {
			iint_minp = (iint_minm+1)%PlotInfo['nint_min']
			ip = 0
		} else {
			iint_minp = iint_minm
		}
		
		for (ib=0; ib<PlotInfo['nbody']; ib++) {

			isegmm = PlotInfo['bodysegm'][ib][iint_minm]
			iom = segm_store * isegmm * 2
						
			if (PlotInfo["InterSegmTimeRev"][ib][iint_minm] > 0) {
				iim = iom + 2*im
			} else {
				iim = iom + 2*(segm_size-im)
			}		

			xlm = PlotInfo['InterSegmSpaceRot'][ib][iint_minm][0][0] * Pos.data[ iim ] + PlotInfo['InterSegmSpaceRot'][ib][iint_minm][0][1] * Pos.data[ iim + 1 ] 
			ylm = PlotInfo['InterSegmSpaceRot'][ib][iint_minm][1][0] * Pos.data[ iim ] + PlotInfo['InterSegmSpaceRot'][ib][iint_minm][1][1] * Pos.data[ iim + 1 ] 

			isegmp = PlotInfo['bodysegm'][ib][iint_minp]
			iop = segm_store * isegmp * 2
	
			if (PlotInfo["InterSegmTimeRev"][ib][iint_minp] > 0) {
				iip = iop + 2*ip
			} else {
				iip = iop + 2*(segm_size-ip)
			}

			xlp = PlotInfo['InterSegmSpaceRot'][ib][iint_minp][0][0] * Pos.data[ iip ] + PlotInfo['InterSegmSpaceRot'][ib][iint_minp][0][1] * Pos.data[ iip + 1 ] 
			ylp = PlotInfo['InterSegmSpaceRot'][ib][iint_minp][1][0] * Pos.data[ iip ] + PlotInfo['InterSegmSpaceRot'][ib][iint_minp][1][1] * Pos.data[ iip + 1 ] 

			xmid = (1-trem) * xlm + trem * xlp - center_x
			ymid = (1-trem) * ylm + trem * ylp - center_y

			xrot_glob = center_x + GlobalRot[0][0] * xmid + GlobalRot[0][1] * ymid
			yrot_glob = center_y + GlobalRot[1][0] * xmid + GlobalRot[1][1] * ymid
			
			p = particles[ib]

			p.lastX = p.x;
			p.lastY = p.y;
			p.x = xrot_glob;
			p.y = yrot_glob;
			
		}

	}

	function speedSliderHandler() {
		setPeriodTime();	
	}
	
	function SpeedMinusClick() {
		var speedSlider = $("#speedSlider");
		var cur_val = speedSlider.slider("value");
		var min_val = speedSlider.slider("option", "min");
		var max_val = speedSlider.slider("option", "max");
		var step_val = speedSlider.slider("option", "step");

		target_val = cur_val - step_val;

		if (target_val > min_val ) {
			speedSlider.slider("value",target_val);
		}

		setPeriodTime();

	}

	function SpeedPlusClick() {
		var speedSlider = $("#speedSlider");
		var cur_val = speedSlider.slider("value");
		var min_val = speedSlider.slider("option", "min");
		var max_val = speedSlider.slider("option", "max");
		var step_val = speedSlider.slider("option", "step");

		target_val = cur_val + step_val;

		if (target_val < max_val ) {
			speedSlider.slider("value",target_val);
		}

		setPeriodTime();

	}

	function setPeriodTime() {

		var slider_value = $("#speedSlider").slider("value")
		var slider_value_rel = slider_value/speed_slider_value_init
		var sliderpow = 3
		var alpha = Math.pow(slider_value_rel,sliderpow)

		var Time_One_Period = Time_One_Period_init / alpha

		var dx = xMax - xMin
		var dy = yMax - yMin
		var distance_ref = Math.sqrt(dx*dx + dy*dy)
		var distance_rel = Max_PathLength / distance_ref

		tInc = 1/(Time_One_Period*FPS_estimation * distance_rel) 

		real_period_estimation = Time_One_Period * distance_rel

		var speedTxt_val = Math.round(100*Math.pow(slider_value,sliderpow));
		speedTxt.innerHTML = "Speed: "+speedTxt_val.toString();

	}

	function FinalizeSetOrbit(DoDrawParticles=true,DoXMinMax=true,setTinc=true) {

		if(DoXMinMax) {

			if (GetPlotInfoChoreoVersion() == "legacy") {
				plotWindow = {
					xMin : PlotInfo["xinf"],
					xMax : PlotInfo["xsup"],
					yMin : PlotInfo["yinf"],
					yMax : PlotInfo["ysup"],
				}
			} else {

				hside = 0.5 * Math.max(PlotInfo["AABB"][1][0]-PlotInfo["AABB"][0][0],PlotInfo["AABB"][1][1]-PlotInfo["AABB"][0][1])

				xmid = 0.5*(PlotInfo["AABB"][1][0]+PlotInfo["AABB"][0][0])
				ymid = 0.5*(PlotInfo["AABB"][1][1]+PlotInfo["AABB"][0][1])
				
				plotWindow = {
					xMin : xmid-hside, 
					xMax : xmid+hside,
					yMin : ymid-hside,
					yMax : ymid+hside,
				}

			}

			setPlotWindow(plotWindow)
		}

		SlideTrailTime()
		makeParticles()

		setParticlePositions(time)
		resetLastPositions()
		setStartPositions()
		
		//if stopped, draw particles in correct place
		if (!running) {
			clearParticleLayer()
			if (DoDrawParticles) {
				drawParticles()
			}
		}
		if (setTinc) {
			setPeriodTime()
		}
		
	}

	function CompleteSetOrbit() {
		
		clearScreen();
		FinalizeSetOrbit();

		if (trajectoriesOn && document.getElementById('checkbox_DisplayLoopOnGalleryLoad').checked){

			if (DoTrailVanish) {
				request = requestAnimationFrame(anim_path_grey)
			} else {
				request = requestAnimationFrame(anim_path)
			}
			
		}

	}

}
