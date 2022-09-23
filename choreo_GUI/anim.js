// Gallery_cache_behavior = {cache: "no-cache"}
Gallery_cache_behavior = {}

var Pos ;
var PlotInfo;

function AjaxGet(foldername){ return $.ajax({ url: foldername})}

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


function trace(message) {
	try {
		console.log(message);
	}
	catch (exception) {
		return;
	}
}

function windowLoadHandler() {
	canvasApp();
}

function canvasApp() {
		
	var particles;

	var numOrbits;

	var tInc;
	var tIncMin, tIncMax;
	var xMin, xMax, yMin, yMax;
	var xPixRate, yPixRate;
	var time;
	var particleRad_mul;
	var bgColor;
	var request;
	var running;
	var trailWidth;
	var trajectoriesOn;
	var colorLookup;
	var defaultParticleColor;
	var staticOrbitColor;
	var staticOrbitWidth;
	var trailColorLookup;
	
	var orbitDrawStartTime;
	var orbitDrawTime;
	var drawingStaticOrbit;
	
	var staticOrbitDrawPointsX;
	var staticOrbitDrawPointsY;
	
	var endPixX;
	var endPixY;
	
	var displayCanvas = document.getElementById("displayCanvas");
	var context = displayCanvas.getContext("2d");
	displayCanvas.addEventListener("FinalizeSetOrbitFromOutsideCanvas", FinalizeSetOrbitFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("StopAnimationFromOutsideCanvas", StopAnimationFromOutsideCanvasHandler, true);
	displayCanvas.addEventListener("StartAnimationFromOutsideCanvas", StartAnimationFromOutsideCanvasHandler, true);
	
	var particleLayerCanvas = document.getElementById("particleLayerCanvas");
	var particleLayerContext = particleLayerCanvas.getContext("2d");
	particleLayerCanvas.addEventListener("click", startStopButtonHandler, true);
	
	var displayWidth = displayCanvas.width;
	var displayHeight = displayCanvas.height;
	
	var startStopButton = document.getElementById("startStopButton");
	startStopButton.addEventListener("click", startStopButtonHandler, true);
	
	var trajectoryButton = document.getElementById("trajectoryButton");
	trajectoryButton.addEventListener("click", trajectoryButtonHandler, true);

	var drawTrajButton = document.getElementById("drawTrajButton");
	drawTrajButton.addEventListener("click", drawTrajButtonHandler, true);
	
	// var AddOrbitButton = document.getElementById("AddOrbitButton");
	// AddOrbitButton.addEventListener("click", AddOrbitButtonHandler, true);
	// 
	// var RemoveOrbitButton = document.getElementById("RemoveOrbitButton");
	// RemoveOrbitButton.addEventListener("click", RemoveOrbitButtonHandler, true);
	
	var btnNextOrbit = document.getElementById("btnNextOrbit");
	btnNextOrbit.addEventListener("click", nextOrbit, true);
	
	var btnPrevOrbit = document.getElementById("btnPrevOrbit");
	btnPrevOrbit.addEventListener("click", prevOrbit, true);

	var AllPosFilenames = [];
	var AllPlotInfoFilenames = [];
	var AllPos = [];
	var AllPlotInfo = [];
	var AllGalleryNames = [];


	//requestAnimationFrame shim for multiple browser compatibility by Eric Möller,
	//http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating
	//For an alternate version, also see http://www.paulirish.com/2011/requestanimationframe-for-smart-animating/.checked 
	(function() {
		var lastTime = 0;
		var vendors = ['ms', 'moz', 'webkit', 'o'];
		for(var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
			window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
			window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame'] 
									   || window[vendors[x]+'CancelRequestAnimationFrame'];
		}
	 
		if (!window.requestAnimationFrame)
			window.requestAnimationFrame = function(callback, element) {
				var currTime = new Date().getTime();
				var timeToCall = Math.max(0, 16 - (currTime - lastTime));
				var id = window.setTimeout(function() { callback(currTime + timeToCall); }, 
				  timeToCall);
				lastTime = currTime + timeToCall;
				return id;
			};
	 
		if (!window.cancelAnimationFrame)
			window.cancelAnimationFrame = function(id) {
				clearTimeout(id);
			};
	}());
	
	init();
	
	async function init() {
		// Particle radius
		particleRad_mul = 5.5;

		// with of particle trail
		trailWidth = 2;

		// Background color
		// bgColor = "#F1F1F1";
		bgColor = "rgb(241,241,241)";

		// Speed of fade inversly prop to alpha channel here
		fadeScreenColor = "rgba(255,255,255,0.01)";
		// fadeScreenColor = "rgba(0,0,0,0)";

		// Color of orbits below
		// staticOrbitColor = "rgba(130,180,270,0.3)";
		// staticOrbitColor = "rgba(130,180,270,0.2)";
		staticOrbitColor = "rgba(200,200,200,0.5)";
		// staticOrbitColor = "rgba(200,200,200,0.0)";
		// staticOrbitColor = "rgba(255,0,255,0.8)"; //TESTING

		// Width of orbits below
		staticOrbitWidth = trailWidth;

		// defaults when colors are not defined
		defaultParticleColor = "#ee6600";
		defaultTrailColor = "#dd5500";
		
		setColorLookupList();

		trajectoriesOn = true;
		drawingStaticOrbit = true;
		
		//jquery ui elements
		//speed slider
		$("#speedSlider").slider({
		  value: 0.33,
		  orientation: "horizontal",
		  range: "min",
		  max: 1,
		  step: 0.005,
		  slide: speedSliderHandler,
		  change: speedSliderHandler,
		  animate: true
		});
		
		tIncMin = 0.0001;
		tIncMax = 0.01;

		orbitDrawStartTime = orbitDrawTime = time = 0;

		// Load the static gallery
		LoadGallery();

		// startAnimation();

	}
	
	function setColorLookupList() {
		colorLookup = ["#ff7006","#50ce4d","#a253c4","#ef1010","#25b5bc","#E86A96","#edc832","#ad6530","#00773f","#d6d6d6"];
		
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
	
	function UnselectOrbit(){
		Index = $('input[name=orbitGroup]:checked').index('input[name=orbitGroup]');
		if (Index > -1) {
			orbitGroups = $('input[name=orbitGroup]')
			mylabel = orbitGroups[Index].getAttribute("mylabel")
			thelabel=$("label[id="+mylabel+"]")["0"];			
			thelabel.classList.remove('w3-red');
			thelabel.classList.add('w3-light-grey');

			orbitGroups[Index].checked = false;
		}
	}

	function incrementOrbit(inc) {
		//find out what is checked
		oldIndex = $('input[name=orbitGroup]:checked').index('input[name=orbitGroup]');
		currentIndex = (oldIndex + inc + numOrbits) % numOrbits; // negative number handling
		
		orbitGroups = $('input[name=orbitGroup]')
		// make old grey
		mylabel = orbitGroups[oldIndex].getAttribute("mylabel")
		thelabel=$("label[id="+mylabel+"]")["0"];			
		thelabel.classList.remove('w3-red');
		thelabel.classList.add('w3-light-grey');
		// make new red
		mylabel = orbitGroups[currentIndex].getAttribute("mylabel")
		thelabel=$("label[id="+mylabel+"]")["0"];			
		thelabel.classList.add('w3-red');
		thelabel.classList.remove('w3-light-grey');

		//make new selection
		orbitGroups[currentIndex].checked = true;
		//how much is currently scrolled:
		var currentScroll = $('#radioContainer').scrollTop();
		//the current position of the selected radio button:
		var rowPos = $('input[name=orbitGroup]:checked').position();
		//rollTop sets how many pixels of area to be above viewable area:
		var scrollAmount = currentScroll + rowPos.top;
		//stop any currently running animations:
		$('#radioContainer').stop();
		//animate scroll:
		$('#radioContainer').animate({scrollTop:scrollAmount + "px"});
		
		current_value = orbitGroups[currentIndex].getAttribute("value")

		//set orbit
		setOrbit(current_value);
	}
	
	function nextOrbit(evt) {
		incrementOrbit(1);
	}
	function prevOrbit(evt) {
		incrementOrbit(-1);
	}
	
	function setPlotWindow(windowObject) {
		xMin = windowObject.xMin;
		xMax = windowObject.xMax;
		yMin = windowObject.yMin;
		yMax = windowObject.yMax;
		xPixRate = displayWidth/(xMax - xMin);
		yPixRate = displayHeight/(yMin - yMax);
	}

	function RemoveOrbit(i_remove) {

		if (numOrbits > 1) {

			Checked_idx = $('input[name=orbitGroup]:checked').index('input[name=orbitGroup]');
			
			if (i_remove == Checked_idx) {

				if (i_remove == (numOrbits-1)) {
					incrementOrbit(-1);
				} else {	
					incrementOrbit(1);
				}

			}

			$('input[name=orbitGroup]:eq('+i_remove+')').remove();
			$('.radioLabel:eq('+i_remove+')').remove();
			
			numOrbits = numOrbits - 1

		}
	
	}

	function AddNewOrbit(orbitRadio,i) {

		numOrbits = numOrbits + 1;

		//radio button
		var input = document.createElement('input');
		input.type = "radio";
		input.value = i;
		input.id = "radio"+i;
		if (i == 0) {
			input.checked = "checked";
		}
		input.name = "orbitGroup";
		input.className = "radioInvisible";

		//label for the button
		var label = document.createElement('label');
		label.id = "label"+i;
		label.setAttribute("for",input.id);
		input.setAttribute("mylabel",label.id)
		label.className = "radioLabel w3-button w3-hover-pale-red w3-light-grey ";
		label.innerHTML = AllGalleryNames[i];

		input.addEventListener("change",function() {

			$("label").filter(".w3-red").each(function(j, obj) {
				obj.classList.remove('w3-red');
				obj.classList.add('w3-light-grey');
			});

			mylabel = this.getAttribute("mylabel");
			thelabel=$("label[id="+mylabel+"]")["0"];			
			thelabel.classList.add('w3-red');
			thelabel.classList.remove('w3-light-grey');

			setOrbit(this.value);

		});
		
		//add to DOM
		orbitRadio.appendChild(input);
		orbitRadio.appendChild(label);
		
	}
	
	function startAnimation() {
		running = true;
		(function animloop(){
		  request = requestAnimationFrame(animloop);
		  onTimer();
		})();
	}

	function stopAnimation() {
		running = false;
		cancelAnimationFrame(request);
	}
	
	function startStopButtonHandler(e) {
		if (running) {
			stopAnimation();
			running = false;
			startStopButton.textContent = "Start";
		}
		else {
			startAnimation();
			running = true;
			startStopButton.textContent = "Stop";
		}
	}
	
	function trajectoryButtonHandler(e) {
		if (trajectoriesOn) {
			trajectoriesOn = false;
			drawingStaticOrbit = false;
			trajectoryButton.textContent ="Show trajectories";
			clearScreen();
		}
		else {
			orbitDrawStartTime = orbitDrawTime = time;
			drawingStaticOrbit = true;
			setStartPositions();
			trajectoriesOn = true;
			trajectoryButton.textContent = "Hide trajectories";
		}
	}

	function drawTrajButtonHandler(e) {
		drawAllSegments();
	}

	function StopAnimationFromOutsideCanvasHandler(e) {
		stopAnimation();
		running = false;
	}

	function StartAnimationFromOutsideCanvasHandler(e) {
		startAnimation();
		running = true;
	}

	function FinalizeSetOrbitFromOutsideCanvasHandler(e) {
		UnselectOrbit();
		clearScreen();
		FinalizeSetOrbit() ;
	}

	function AddOrbitButtonHandler(e) {
		i = numOrbits;
		var orbitRadio = document.getElementById("orbitRadio");
		AddNewOrbit(orbitRadio,jsonData,i)
	}

	function RemoveOrbitButtonHandler(e) {

		i_remove = $('input[name=orbitGroup]:checked').index('input[name=orbitGroup]');

		RemoveOrbit(i_remove)
	}

	function onTimer() {

		if (trajectoriesOn) {
			//fade
			context.fillStyle = fadeScreenColor;
			context.fillRect(0,0,displayWidth,displayHeight);
		}
		
		//clear particle layer
		clearParticleLayer();
		
		if (drawingStaticOrbit) {
			orbitDrawTime += tInc;
			// console.log(orbitDrawTime);

			if (orbitDrawTime > orbitDrawStartTime + 1) {
				//stop drawing orbit
				drawingStaticOrbit = false;
				//draw last segments
				// drawLastSegments();
			}
		}		
		
		time = (time + tInc) % 1;
		
		//update particles
		setParticlePositions(time);
		
		//draw particles
		drawParticles();
	}
	
	function clearScreen() {
		context.fillStyle = bgColor;
		context.fillRect(0,0,displayWidth,displayHeight);
	}
	
	function clearParticleLayer() {
		particleLayerContext.clearRect(0,0,displayWidth+1,displayHeight+1);
	}
	
	function makeParticles(colors) {

		particles = [];
		
		for (var i = 0; i<PlotInfo['nbody']; i++) {
			var color;
			var trailColor;
			if (i<colors.length) {
				color = colorLookup[colors[i]];
				trailColor = trailColorLookup[colors[i]];
			}
			else {
				color = defaultParticleColor;
				trailColor = defaultTrailColor;
			}
					
			var p = {
					x: 0,
					y: 0,
					lastX: 0,
					lastY: 0,
					color: color,
					trailColor: trailColor,
					particleRad: Math.sqrt(PlotInfo["mass"][i]) * particleRad_mul
			}
			particles.push(p);
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
	
	function drawLastSegments() {
		var i;
		var p;
		var pixX;
		var pixY;

		for (i = 0; i < PlotInfo["nbody"]; i++) {
			p = particles[i];
			pixX = xPixRate*(p.x - xMin);
			pixY = yPixRate*(p.y - yMax);

			//trail
			context.strokeStyle = staticOrbitColor;
			context.lineWidth = staticOrbitWidth;
			context.beginPath();
			context.moveTo(staticOrbitDrawPointsX[i],staticOrbitDrawPointsY[i]);
			context.lineTo(pixX, pixY);
			context.stroke();

			staticOrbitDrawPointsX[i] = pixX;
			staticOrbitDrawPointsY[i] = pixY;

		}

	}

	function drawAllSegments() {
		
		clearScreen();
		setStartPositions();
		
		n_strokes = Math.floor((1 / tInc)) + 1 ;

		// Setup
		for (i = 0; i < PlotInfo["nbody"]; i++) {
			p = particles[i];
			pixX = xPixRate*(p.x - xMin);
			pixY = yPixRate*(p.y - yMax);

			staticOrbitDrawPointsX[i] = pixX;
			staticOrbitDrawPointsY[i] = pixY;
		}

		for (i_stroke = 0; i_stroke < n_strokes; i_stroke++) {

			time = (time + tInc) % (1);

			drawLastSegments();
			setParticlePositions(time);
		
		}

	}
	
	function drawParticles() {
		var i;
		var len;
		var pixX;
		var pixY;
		var lastPixX;
		var lastPixY;
		var p;
		var dx;
		var dy;
		
		context.lineCap = "round";
		
		len = particles.length;
		for (i = 0; i < len; i++) {
			p = particles[i];
			pixX = xPixRate*(p.x - xMin);
			pixY = yPixRate*(p.y - yMax);
			lastPixX = xPixRate*(p.lastX - xMin);
			lastPixY = yPixRate*(p.lastY - yMax);
						
			//particle
			particleLayerContext.strokeStyle = "rgba(0,0,0,0.5)"
			particleLayerContext.lineWidth = 2;
			particleLayerContext.fillStyle = p.color;
			particleLayerContext.beginPath();
			particleLayerContext.arc(pixX,pixY,p.particleRad,0,Math.PI*2,false);
			particleLayerContext.closePath();
			particleLayerContext.fill();
			particleLayerContext.stroke();
			
			if (trajectoriesOn) {
				//trail
				context.strokeStyle = p.trailColor;
				context.lineWidth = trailWidth;
				context.beginPath();
				context.moveTo(lastPixX,lastPixY);
				context.lineTo(pixX, pixY);
				context.stroke();

			}
		}

	}
	
	function setParticlePositions(t) {

		var n_pos = Pos.shape[2]

		var xlm=0,xlp=0,ylm=0,ylp=0;
		var xmid,ymid;
		var xrot,yrot;

		var il,ib,ilb,nlb;
		var tb;

		var im,ip,tbn,trem;
		
		im = Math.floor(t*n_pos)
		ip = (im+1) % n_pos

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

				xrot = PlotInfo['SpaceRotsUn'][il][ilb][0][0] * xmid + PlotInfo['SpaceRotsUn'][il][ilb][0][1] * ymid 
				yrot = PlotInfo['SpaceRotsUn'][il][ilb][1][0] * xmid + PlotInfo['SpaceRotsUn'][il][ilb][1][1] * ymid 
				
				particles[ib].lastX = particles[ib].x;
				particles[ib].lastY = particles[ib].y;
				particles[ib].x = xrot;
				particles[ib].y = yrot;
				
			}
		
		}

	}

	function speedSliderHandler() {
		setTInc();	
	}
	
	function setTInc() {

		var slider_value = $("#speedSlider").slider("value");
		var sliderpow = 2;
		var alpha = Math.pow(slider_value,sliderpow);

		tInc = tIncMin + (tIncMax - tIncMin)*alpha;
	}

	function FinalizeSetOrbit() {
		
		plotWindow = {
			xMin : PlotInfo["xinf"],
			xMax : PlotInfo["xsup"],
			yMin : PlotInfo["yinf"],
			yMax : PlotInfo["ysup"],
		}

		setPlotWindow(plotWindow);
		
		var colors;

		if (PlotInfo['nbody'] < colorLookup.length) {
			//if fewer than color list, default will be to do different colors.
			colors = [];
			for (var i = 0; i < PlotInfo['nbody']; i++) {
				colors.push(i);
			}
		}
		else {
			//if more than color list, set to empty array,then default will be to make all same color.
			colors = [];
		}

		makeParticles(colors);
		clearScreen();
				
		time = 0;
		
		if (trajectoriesOn) {
			drawingStaticOrbit = true;
			orbitDrawStartTime = orbitDrawTime = time;
		}
		else {
			drawingStaticOrbit = false;
		}
		
		setParticlePositions(0);
		resetLastPositions();
		setStartPositions();
		
		//if stopped, draw particles in correct place
		if (!running) {
			clearParticleLayer();
			drawParticles();	
		}
		setTInc();
		
	}

	function setOrbit(orbitIndex) {
		
		Pos = AllPos[orbitIndex];
		PlotInfo = AllPlotInfo[orbitIndex];

		FinalizeSetOrbit();

	}

	async function LoadGallery() {
			
		var gallery_filename = "gallery_descriptor.json"

		var Gallery_description;

		await fetch(gallery_filename,Gallery_cache_behavior)
			.then(response => response.text())
			.then(data => {
				Gallery_description = JSON.parse(data);
			})

		for (const [name, path] of Object.entries(Gallery_description)) {

				AllPosFilenames.push(path+'.npy');
				AllPlotInfoFilenames.push(path+'.json');
				AllGalleryNames.push(name);

		}


		n_init_gallery_orbits = AllPosFilenames.length;
		numOrbits = 0;
		var orbitRadio = document.getElementById("orbitRadio");

		for (var i = 0; i < n_init_gallery_orbits; i++) {

			AddNewOrbit(orbitRadio,i);

		}

		AllPos = new Array(n_init_gallery_orbits);
		AllPlotInfo = new Array(n_init_gallery_orbits);

		// Load all files asynchronously, keeping promises

		for (var i = 0; i < n_init_gallery_orbits; i++) {
			
			let npyjs_obj = new npyjs();
// 
			let finished_npy = 
				npyjs_obj.load(AllPosFilenames[i])
				.then((res) => {
					AllPos[i] = res;
				});

			let finished_json = 
				fetch(AllPlotInfoFilenames[i],Gallery_cache_behavior)
				.then(response => response.text())
				.then(data => {
					AllPlotInfo[i] = JSON.parse(data);
				});

			await Promise.all([finished_npy ,finished_json ])

			if (i==0) {

				// await Promise.all([finished_npy ,finished_json ])

				$('label:first', "#orbitRadio").removeClass('w3-light-grey').addClass('w3-red');
				setOrbit(0);
				
				startAnimation();

			}
		}
	}
}
