function init() {
	//Custom Selects
	$("select").select2({
		dropdownCssClass: 'dropdown-inverse'
	});
	$('.input-group').on('focus', '.form-control', function () {
		return $(this).closest('.input-group, .form-group').addClass('focus');
	}).on('blur', '.form-control', function () {
		return $(this).closest('.input-group, .form-group').removeClass('focus');
	});
	$(".btn-group").on('click', "a", function () {
		return $(this).siblings().removeClass("active").end().addClass("active");
	});
}

function doSearch() {
	let mgf = $('#mgf').val();
	// console.log(mgf);
	
	$('#info, #loadErr, #complete, #rst').slideUp(100);
	$('#loading').slideDown(300);

	jQuery.post("/api/denovo", {'mgf': mgf})
	.done(function(data) {
		console.log(data);
		try {			
			const rst = JSON.parse(data.replaceAll("'", "\""));

			$('#pep').html(rst.pep);
			$('#score').html(rst.score);

			let ps = '[';

			for (let s of rst.pscore) {
				ps += s.toFixed(3) + ", ";
			}

			$('#pscore').html(ps + ']');

			$('#info, #loadErr, #loading').slideUp(200);

			$('#rst, #complete').slideDown(400);
			
			$('body, html').animate({
				scrollTop: $('#complete')[0].offsetTop - 150
			}, 1000);
		}
		catch (err) {
			console.log(err);

			$('#info, #loading, #complete, #rst').slideUp(200);

			$('#loadErr').slideDown(400);

			$('body, html').animate({
				scrollTop: $('#loadErr')[0].offsetTop - 150
			}, 1000);
		}
	})
	.fail(function() {
		$('#info, #loading, #complete, #rst').slideUp(200);

		$('#loadErr').slideDown(400);

		$('body, html').animate({
			scrollTop: $('#loadErr')[0].offsetTop - 100
		}, 1000);
	})
}

init();

$('#complete, #loading, #loadErr, #rst').removeClass('hidden').slideUp(0);

$('#search').on('click', function (event) {
	return doSearch();
});
