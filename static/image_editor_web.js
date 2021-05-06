window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox = $('#imagebox')
		input = $('#imageinput')[0]
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('image' , input.files[0]);
			$.ajax({
				url: "http://localhost:5000/editImage", // fix this to your liking
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				except: "100-continue",
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					// alert("hello"); // if it's failing on actual server check your server FIREWALL + SET UP CORS
					bytestring = data['status']
					image = bytestring.split('\'')[1]
					returnimagebox = $('#returnimagebox')
					returnimagebox.attr('src' , 'data:image/jpeg;base64,'+image)
					returnimagebox.height(window.height/2);
			        returnimagebox.width(window.width/2);

			        // plot histograms
			        hist_orig = data['hist_orig']
			        hist_edit = data['hist_edit']

			        toHist(hist_orig[0], hist_edit[0], 'red', 'rHist')
                    toHist(hist_orig[1], hist_edit[1], 'green', 'gHist')
                    toHist(hist_orig[2], hist_edit[2], 'blue', 'bHist')
				}
			});
		}
	});
};

function toHist(data1, data2, color, spot){
    var original = {
      x: data1,
      name: 'Original',
      type: "histogram",
      opacity: 0.25,
      marker: {
         color: color,
      },
    };
    var edited = {
      x: data2,
      name: "Edited",
      type: "histogram",
      opacity: 0.75,
      marker: {
         color: color,
      },
    };

    var data = [original, edited];
    var layout = {barmode: "overlay"};
    Plotly.newPlot(spot, data, layout);
}

function readUrl(input){
	imagebox = $('#imagebox')
	console.log("evoked readUrl")
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){
			// console.log(e)

			imagebox.attr('src',e.target.result);
			imagebox.height(window.height/2);
			imagebox.width(window.width/2);
		}
		reader.readAsDataURL(input.files[0]);
	}


}